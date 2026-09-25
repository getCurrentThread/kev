"""kev.vision parity on real weights: does a checkpoint score the same through load_vision, and do image rows equal the
base's own multimodal forward?

    uv run python scripts/vision_parity.py --run jaredpalmer/kev-0.8b --device cpu --suite evals/smoke-v1
    uv run python scripts/vision_parity.py --run jaredpalmer/kev-4b --device cuda --bf16      # adapter unmerged, bf16

1. Text records: load() vs load_vision() probabilities (max |dp|, argmax flips); the same adapter on the same language
   model. Two built-in records, plus the development partition of --suite through the serving path (kev.data.materialize).
2. Image records (synthetic pictures of three sizes, no download): VisionDecisionModel.forward vs the pointer head read
   off Qwen3_5Model.forward(input_ids, pixel_values, image_grid_thw, mm_token_type_ids) for every row (max |dlogit|,
   max |dp|, argmax flips), plus the image-token counts the pictures became.
"""
import argparse

import numpy as np
import torch
from PIL import Image

from kev.checkpoint import Checkpoint, LoadOptions
from kev.data import materialize
from kev.device import empty_cache
from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE, rows_of
from kev.suite import load_split

RECORDS = [
    {"state": "Order 4411 arrived late and the box was crushed. Two charges appear on the card.",
     "questions": [{"instr": "Is there a billing problem?", "options": ["yes", "no"], "label": 0},
                   {"instr": "Which team should handle this?", "options": ["returns", "shipping", "billing", "other"], "label": 2}]},
    {"state": "", "questions": [{"instr": "What is the main colour of the picture?", "options": ["red", "green", "blue", "grey"], "label": 0}]},
]
SIZES = [(96, 128), (480, 640), (720, 960)]   # height, width


def picture(h, w, seed):
    """A gradient with a little noise: enough structure for the vision tower, no download."""
    y, x = np.mgrid[0:h, 0:w]
    rgb = np.stack([x * 255 // w, y * 255 // h, (x + y) * 255 // (h + w)], -1)
    return Image.fromarray(np.clip(rgb + np.random.default_rng(seed).integers(-20, 20, rgb.shape), 0, 255).astype("uint8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-0.8b")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bf16", action="store_true", help="LoadOptions(dtype=bf16, merge=False) instead of LoadOptions() (fp32 merged, "
                    "except for checkpoints trained on a bf16 backbone, such as Kev-27B, which load bf16 unmerged either way)")
    ap.add_argument("--suite", help="also compare the text path on this frozen suite's development records (e.g. evals/smoke-v1)")
    a = ap.parse_args()
    opts = LoadOptions(dtype=torch.bfloat16, merge=False) if a.bf16 else LoadOptions()
    ctx = {"max_state": SERVE_MAX_STATE, "max_branch": SERVE_MAX_BRANCH}
    ck = Checkpoint(a.run)
    texts = RECORDS + ([materialize(r) for r in load_split(a.suite, "development")] if a.suite else [])

    tok, text = ck.load(a.device, opts)
    with torch.no_grad():
        expected = [p for r in texts for p in text.probs(text.encode(tok, r, **ctx))]   # one tensor per question
    del text; empty_cache(a.device)
    tok, m = ck.load_vision(a.device, opts)
    with torch.no_grad():
        got = [p for r in texts for p in m.probs(m.encode(tok, r, **ctx))]
        text_dp = max((x - y).abs().max().item() for x, y in zip(expected, got))
        text_flips = sum(int(x.argmax() != y.argmax()) for x, y in zip(expected, got))
        dz, dp, flips, rows, tokens = 0.0, 0.0, 0, 0, []
        for seed, (h, w) in enumerate(SIZES):
            for rec in RECORDS:
                enc = m.encode(tok, rec, image=picture(h, w, seed), **ctx)
                tokens.append(sum(t == m.image_pad for t in enc["ids"]))
                S, _, branches = rows_of(enc)
                for z, r in zip(m.forward(enc), branches):
                    ids = torch.tensor([S + r["ids"]], device=m.device)
                    hid = m.vlm(input_ids=ids, pixel_values=enc["pixel_values"].to(m.device), image_grid_thw=enc["image_grid_thw"].to(m.device),
                                mm_token_type_ids=(ids == m.image_pad).int()).last_hidden_state[0].float()
                    ref = m.head(hid[len(S) + r["decide"]], hid[torch.tensor([len(S) + o for o in r["opts"]], device=m.device)])
                    dz = max(dz, (z - ref).abs().max().item())
                    dp = max(dp, (z.softmax(-1) - ref.softmax(-1)).abs().max().item())
                    flips += int(z.argmax() != ref.argmax()); rows += 1
    path = f"{m.dtype}, adapter {'unmerged' if hasattr(m.lm, 'peft_config') else 'merged'}"   # what loaded, not what was asked
    n_suite = len(texts) - len(RECORDS)
    print(f"{a.run} on {a.device} ({path})")
    print(f"  text, load vs load_vision: max |dp| {text_dp:.2g}, {text_flips} argmax flips "
          f"({len(texts)} records = {len(RECORDS)} built-in + {n_suite} {a.suite or ''} development; {len(got)} questions)")
    print(f"  image rows vs Qwen3_5Model.forward: max |dlogit| {dz:.2g}, max |dp| {dp:.2g}, {flips} argmax flips "
          f"({rows} rows in {len(tokens)} records; image tokens {sorted(set(tokens))})")


if __name__ == "__main__":
    main()
