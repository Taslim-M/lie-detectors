"""Download the factual-statement CSVs used to train the probe, pinned to exact commits.

Sources (paper Sec. 5.2-5.3):
  * cities, sp_en_trans (+ negated): Marks & Tegmark (2024), saprmarks/geometry-of-truth
  * element_symb (+ negated): Azaria & Mitchell (2023), quality-filtered by Buerger et al. (2024),
    sciai-lab/Truth_is_Universal

Each file is checked against its git blob SHA-1, so the bytes match the pinned commit exactly.

Usage: python scripts/fetch_probe_sources.py [out_dir]   (default: data/probe_sources)
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

MARKS = ("saprmarks/geometry-of-truth", "5d1c630c44f7e50bda7ad86d601ccadf9abc5ddb")
BUERGER = ("sciai-lab/Truth_is_Universal", "605ef00514415deb4806969a172f6e13e0798df7")

FILES = {  # name: (repo, commit, git blob sha)
    "cities.csv": (*MARKS, "333224e86f2eb81e95cc54c5c0fd8a7556c698b9"),
    "neg_cities.csv": (*MARKS, "a6f71db4a309ae9cf586431bc30bb7076e351add"),
    "sp_en_trans.csv": (*MARKS, "f99aa08658917c9000b6db1598458fa29ae4aeea"),
    "neg_sp_en_trans.csv": (*MARKS, "60dd635566cec80004d95cf65a5138da00b4788e"),
    "element_symb.csv": (*BUERGER, "0b5c8f71832798167ea84b1aa1f0ee91e7e7bc31"),
    "neg_element_symb.csv": (*BUERGER, "c456bedc56acb519aee7482bde1dd76c35d0ef4a"),
}


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (repo, commit, blob) in FILES.items():
        dest = out_dir / name
        if dest.exists() and git_blob_sha(dest.read_bytes()) == blob:
            print(f"ok    {name}")
            continue
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/datasets/{name}"
        data = urllib.request.urlopen(url).read()
        if git_blob_sha(data) != blob:
            raise SystemExit(f"blob SHA mismatch for {name} from {url}")
        dest.write_bytes(data)
        print(f"saved {name} ({len(data):,} bytes) from {repo}@{commit[:7]}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data" / "probe_sources")
