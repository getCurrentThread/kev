"""An image as the start of the state, on the base's own vision tower.

Kev's bases (Qwen3.5-*-Base, and Qwen3.8-27B under Kev-27B) are vision-language models, but DecisionModel loads only
their text backbone. The adapter and the pointer head live entirely in that backbone, so they apply unchanged to the same
module inside the full model. VisionDecisionModel loads the full model (AutoModel: vision tower + language model, no vocab
head) and puts the image tokens right after the <state> delimiter:

    <state> <vision_start> <image_pad> x N <vision_end> state text <q> ... <opt> ... </opt> ... <decide>

The question side is kev.model.encode() unchanged. Each question runs as its own causal row (the row form every hybrid
base uses), the vision tower runs once per record, and positions come from the base's multimodal rotary positions
(M-RoPE: the image tokens get 2-D grid positions, text after them continues from the image). A record without an image
takes DecisionModel's path unchanged.

    tok, m = Checkpoint("jaredpalmer/kev-4b").load_vision("cuda", LoadOptions(dtype=torch.bfloat16))
    enc = m.encode(tok, {"state": "", "questions": [{"instr": "What is on the table?", "options": ["a cup", "a book"], "label": 0}]},
                   image=ImageOps.exif_transpose(Image.open("photo.jpg")), max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
    probs = m.probs(enc)                            # one tensor per question

No checkpoint was trained with images: this reuses the base's image understanding through an adapter that only saw text,
and the temperature in head.pt was fitted on text states. scripts/vision_parity.py has checked Kev-0.8B and Kev-4B;
Kev-27B (trained on a bf16 backbone, so it loads bf16 with the adapter unmerged here) has not been run with images. An
image becomes about pixels / 1,024 tokens (each side rounded to a multiple of 32; at least 64, the processor scales small
pictures up) on top of max_state. `m.processor.size` ({"shortest_edge", "longest_edge"} in total pixels) sets the bounds,
capped here at MAX_IMAGE_PIXELS. The image processor is transformers' Pillow backend, so it needs Pillow and not
torchvision, and it does not apply EXIF orientation.
"""
import torch
from transformers import AutoModel
from transformers.models.auto.image_processing_auto import AutoImageProcessor   # the top-level name requires torchvision

from .model import OPT_NONE, DecisionModel, rows_of

# the bases' own processor allows 16.7 MP, ~16k tokens per picture (a 12 MP phone photo is 11,844); cap at ~1,024
MAX_IMAGE_PIXELS = 1024 * 1024


def splice_image(enc, n_image_tokens, vision_start, image_pad, vision_end):
    """encode() output with an image block inserted right after the <state> delimiter: the block belongs to the state
    (segment 0), shifts every later index by its length, and keeps positions contiguous (the model replaces them with
    M-RoPE positions from the ids)."""
    block = [vision_start] + [image_pad] * n_image_tokens + [vision_end]
    k = len(block)
    return {**enc, "ids": enc["ids"][:1] + block + enc["ids"][1:],
            "seg": [0] * (k + 1) + enc["seg"][1:],
            "pos": list(range(k + 1)) + [p + k for p in enc["pos"][1:]],
            "opt": [OPT_NONE] * (k + 1) + enc["opt"][1:],
            "decide_idx": [d + k for d in enc["decide_idx"]],
            "opt_idx": [[o + k for o in oi] for oi in enc["opt_idx"]]}


class VisionDecisionModel(DecisionModel):
    def __init__(self, name, tok, device, revision=None, **kw):
        super().__init__(name, tok, device, revision=revision, **kw)
        if not self.hybrid: raise ValueError(f"{name}: images use the row form, which Kev runs on the hybrid (Qwen3.5) bases only")
        cfg = self.vlm.config
        self.image_pad, self.vision_start, self.vision_end = cfg.image_token_id, cfg.vision_start_token_id, cfg.vision_end_token_id
        self.spatial_merge = cfg.vision_config.spatial_merge_size
        self.processor = AutoImageProcessor.from_pretrained(name, revision=revision, backend="pil")   # same pixels with or without torchvision
        size = self.processor.size
        self.processor.size = {"shortest_edge": size.shortest_edge, "longest_edge": min(size.longest_edge, MAX_IMAGE_PIXELS)}

    def backbone(self, name, **kw):
        """The language model inside the full vision-language model: the module DecisionModel.backbone loads on its own,
        so the adapter's keys match it unchanged."""
        self.vlm = AutoModel.from_pretrained(name, **kw)
        if not hasattr(self.vlm, "get_image_features"): raise ValueError(f"{name} has no vision tower")
        return self.vlm.language_model

    def encode(self, tok, rec, image=None, **kw):
        """DecisionModel.encode, plus `image` (one picture, e.g. a PIL image) at the start of the state. max_state limits
        the state text; the image tokens come on top of it."""
        enc = super().encode(tok, rec, **kw)
        if image is None: return enc
        pix = self.processor(images=[image], return_tensors="pt")
        if len(pix["image_grid_thw"]) != 1: raise ValueError(f"one image per record, got {len(pix['image_grid_thw'])}")
        n = int(pix["image_grid_thw"][0].prod()) // self.spatial_merge ** 2
        return {**splice_image(enc, n, self.vision_start, self.image_pad, self.vision_end),
                "pixel_values": pix["pixel_values"], "image_grid_thw": pix["image_grid_thw"]}

    def forward_rows_batch(self, encs):
        if not any("image_grid_thw" in e for e in encs): return super().forward_rows_batch(encs)
        return [self._image_rows(e) if "image_grid_thw" in e else super().forward_rows_batch([e])[0] for e in encs]

    def _image_rows(self, enc):
        """Logits per question of one record with an image: the rows of forward_rows_batch (state, image included, + one
        branch), with the image features computed once and the backbone inputs of each chunk built by the base's rules."""
        grid = enc["image_grid_thw"].to(self.device)
        image = self.vlm.get_image_features(enc["pixel_values"].to(self.device), grid, return_dict=True).pooler_output[0]

        def inputs(ids, att):   # the features in every row's <image_pad> slots; M-RoPE positions from the ids
            is_image = ids == self.image_pad
            emb = self.vlm.get_input_embeddings()(ids)
            emb = emb.masked_scatter(is_image[..., None].expand_as(emb), image.to(emb.dtype).repeat(len(ids), 1))
            pos, _ = self.vlm.get_rope_index(ids, mm_token_type_ids=is_image.int(), image_grid_thw=grid.repeat(len(ids), 1), attention_mask=att)
            return {"inputs_embeds": emb, "position_ids": pos}

        S, Sp, branches = rows_of(enc)
        hs = self._rows_hidden([(S + r["ids"], Sp + r["pos"]) for r in branches], inputs=inputs)
        return [self.head(h[len(S) + r["decide"]], h[torch.tensor([len(S) + o for o in r["opts"]], device=self.device)]) for h, r in zip(hs, branches)]

    def prefix(self, enc):
        if "image_grid_thw" in enc: raise NotImplementedError("the state-prefix cache does not carry images; use probs()")
        return super().prefix(enc)

    def probs_with_prefix(self, enc, prefix):
        if "image_grid_thw" in enc: raise NotImplementedError("the state-prefix cache does not carry images; use probs()")
        return super().probs_with_prefix(enc, prefix)
