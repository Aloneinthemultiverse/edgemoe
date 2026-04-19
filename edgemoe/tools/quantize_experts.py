"""Requantize an already-split expert directory to a different precision.

Typical use: 4-bit → 1.58-bit ternary. Reads each expert file, unpacks
tensors, applies BitNet ternary quantization, rewrites the file, and
updates manifest.json in place.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import torch

from edgemoe.quantization.adaptive import AdaptiveQuantizer
from edgemoe.quantization.bitnet import BitNetExpertQuantizer
from edgemoe.engine import _rehydrate


def _read_record(path: Path) -> dict:
    raw = path.read_bytes()
    header_len = struct.unpack("<I", raw[:4])[0]
    header = json.loads(raw[4 : 4 + header_len])
    payload = raw[4 + header_len :]
    return _rehydrate(header, payload)


def _write_record(record: dict, path: Path) -> None:
    from tools.split_experts import _serialize_tensor_record
    path.write_bytes(_serialize_tensor_record(record))


def requantize_experts(experts_dir: str, to_bitnet: bool = True) -> None:
    root = Path(experts_dir)
    if (root / "experts").exists():
        root = root / "experts"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    adapt = AdaptiveQuantizer()
    target_label = "bitnet" if to_bitnet else manifest.get("quant_mode", "q4")

    files = sorted(root.glob("L*_E*.bin"))
    for i, path in enumerate(files):
        record = _read_record(path)
        # Reconstruct full-precision weights first.
        full_tensors: dict[str, torch.Tensor] = {}
        keys = sorted(k for k in record if isinstance(record[k], torch.Tensor))
        bases = {k.rsplit(".", 1)[0] for k in keys}
        for base in bases:
            if f"{base}.packed" in record:
                ternary_rec = {
                    "packed": record[f"{base}.packed"],
                    "scale": float(record[f"{base}.scale"][0].item()),
                    "shape": tuple(record.get(f"{base}.shape", ())),
                }
                full_tensors[base] = BitNetExpertQuantizer.dequantize(ternary_rec)
            else:
                q = {"q": record[f"{base}.q"],
                     "scale": record[f"{base}.scale"],
                     "bits": 4,
                     "group_size": 128,
                     "mode": "group_asym"}
                if f"{base}.zp" in record:
                    q["zp"] = record[f"{base}.zp"]
                full_tensors[base] = adapt.dequantize(q)

        # Re-quantise into target format.
        new_record: dict
        if to_bitnet:
            new_record = {"dtype": "ternary"}
            for name, t in full_tensors.items():
                q = BitNetExpertQuantizer.quantize(t)
                new_record[f"{name}.packed"] = q["packed"]
                new_record[f"{name}.scale"] = torch.tensor([float(q["scale"])])
                new_record[f"{name}.shape"] = list(q["shape"])
        else:
            new_record = {"dtype": "group_asym", "mode": "group_asym"}
            for name, t in full_tensors.items():
                q = adapt.quantize(t, bits=4)
                new_record[f"{name}.q"] = q["q"]
                new_record[f"{name}.scale"] = q["scale"]
                if "zp" in q:
                    new_record[f"{name}.zp"] = q["zp"]

        _write_record(new_record, path)
        print(f"[quant] {i+1}/{len(files)}  {path.name}  → {target_label}")

    manifest["quant_mode"] = target_label
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("[quant] Done.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("experts_dir")
    p.add_argument("--bitnet", action="store_true", default=True)
    p.add_argument("--no-bitnet", dest="bitnet", action="store_false")
    args = p.parse_args()
    requantize_experts(args.experts_dir, to_bitnet=args.bitnet)
