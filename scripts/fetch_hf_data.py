"""Download all configs of the maxvonk/anti-factual-personas HF dataset as parquet files.

Usage: python scripts/fetch_hf_data.py [out_dir]   (default: data/anti-factual-personas)
"""
import sys
import urllib.request
from pathlib import Path

REPO = "maxvonk/anti-factual-personas"
CONFIGS = [
    "on-policy-personas-gemma-3-27b-it",
    "on-policy-personas-gemma-4-31b-it",
    "on-policy-personas-llama-70b-3.3",
    "spp-personas-gemma-3-27b-it",
    "spp-personas-gemma-4-31b-it",
    "spp-personas-llama-70b-3.3",
    "confounder-compliance-brackets",
    "confounder-persona-stance-nora",
    "confounder-probability-score-learning-16turn",
]


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for cfg in CONFIGS:
        dest = out_dir / f"{cfg}.parquet"
        if dest.exists():
            print(f"skip {dest}")
            continue
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/data/{cfg}/train.parquet"
        urllib.request.urlretrieve(url, dest)
        print(f"saved {dest} ({dest.stat().st_size:,} bytes)")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data" / "anti-factual-personas")
