# worker_transcribe.py
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import whisper

def get_device(device_choice: str):
    device_choice = device_choice.lower()
    if device_choice == "dml":
        import torch_directml  # must be installed in the same env
        return torch_directml.device()
    if device_choice == "cuda":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", required=True, choices=["cpu", "cuda", "dml"])
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    inp = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)

    # Load + move model
    model = whisper.load_model(args.model)
    try:
        model = model.to(device)
    except Exception:
        # If .to(device) fails for any reason, fall back to cpu
        model = model.to("cpu")
        device = "cpu"

    # Transcribe
    result = model.transcribe(str(inp), fp16=args.fp16, verbose=False)

    # Write output
    txt_path = out_dir / f"{inp.stem}_transcript.txt"
    txt_path.write_text((result.get("text") or "").strip() + "\n", encoding="utf-8")

    # Print a simple “100%” marker for your UI if you want
    print("100%|", file=sys.stderr)

if __name__ == "__main__":
    main()
