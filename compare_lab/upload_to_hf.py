"""Publish Cracked-D + its tokenizer to the Hub for the Streamlit demo.

Everything goes under a compare_lab/ prefix in the same repo the other phases
use, so the deployed app pulls each phase's artefacts from one place.

The checkpoint uploaded is the SLIM export (weights + config, no optimizer
state): 100MB instead of 301MB, which is what keeps the deployed app under
Streamlit Community Cloud's 1GB memory cap. Run export_for_demo.py first.

Requires an existing `huggingface-cli login` session -- this never asks for a
token.

Usage:
    python -m compare_lab.upload_to_hf --arch cracked
"""

import argparse
from pathlib import Path

HF_REPO = "davidmorgado/coconut-mini-llm"
PKG = Path(__file__).resolve().parent
TOKENIZER_DIR = PKG / "data" / "artifacts" / "tokenizer"

# local file -> path in the repo
REMOTE_NAMES = {"cracked": "compare_lab/cracked_d_final.pt",
                "cracked_full": "compare_lab/cracked_d_full_final.pt",
                "sliced": "compare_lab/sliced_d_final.pt"}


def upload(arch: str = "cracked", repo: str = HF_REPO, dry_run: bool = False) -> list[dict]:
    from huggingface_hub import HfApi, whoami

    who = whoami()["name"]  # fails loudly if there is no login session
    ckpt = PKG / "checkpoints" / arch / "finetune_slim.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"{ckpt} missing -- run `python -m compare_lab.export_for_demo --arch {arch}` first")

    plan = [{"local": ckpt, "remote": REMOTE_NAMES[arch]},
            {"local": TOKENIZER_DIR / "vocab.json", "remote": "compare_lab/tokenizer/vocab.json"},
            {"local": TOKENIZER_DIR / "merges.txt", "remote": "compare_lab/tokenizer/merges.txt"}]
    for item in plan:
        if not item["local"].exists():
            raise FileNotFoundError(item["local"])
        item["mb"] = round(item["local"].stat().st_size / 1048576, 1)

    print(f"sesion HF: {who} -> {repo}")
    for item in plan:
        print(f"  {item['local'].name:20} {item['mb']:7.1f} MB -> {item['remote']}")
    if dry_run:
        print("dry-run: no se ha subido nada")
        return plan

    api = HfApi()
    for item in plan:
        api.upload_file(path_or_fileobj=str(item["local"]), path_in_repo=item["remote"],
                        repo_id=repo, repo_type="model")
        print(f"  subido {item['remote']}")
    return plan


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="cracked", choices=sorted(REMOTE_NAMES))
    p.add_argument("--repo", default=HF_REPO)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    upload(args.arch, args.repo, args.dry_run)


if __name__ == "__main__":
    main()
