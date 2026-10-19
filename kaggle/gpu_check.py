"""Smoke test: confirms the Kaggle session has a GPU and can read the data branch."""
import argparse
import json
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--data")
ap.add_argument("--out")
a = ap.parse_args()
info = {"cuda": torch.cuda.is_available(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "parquet_files": len(list(Path(a.data, "cfpb").rglob("*.parquet")))}
print(info)
Path(a.out, "summary.json").write_text(json.dumps(info))
