import argparse
import json
import pathlib
import shutil

from huggingface_hub import hf_hub_download


def download_cosmos_tokenizer(repo_id: str, output_dir: str, force: bool = False) -> dict:
    out = pathlib.Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    filenames = ("encoder.jit", "decoder.jit", "autoencoder.jit", "config.json", "model_config.yaml", "README.md")
    copied = {}
    for filename in filenames:
        target = out / filename
        if target.exists() and not force:
            copied[filename] = str(target)
            continue
        source = pathlib.Path(hf_hub_download(repo_id, filename))
        shutil.copy2(source, target)
        copied[filename] = str(target)
    summary = {
        "repo_id": repo_id,
        "output_dir": str(out),
        "files": copied,
    }
    (out / "download_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Cosmos tokenizer JIT files.")
    parser.add_argument("--repo-id", default="nvidia/Cosmos-Tokenizer-CI16x16")
    parser.add_argument("--output-dir", default="data/checkpoints/cosmos_tokenizer/Cosmos-Tokenizer-CI16x16")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(download_cosmos_tokenizer(args.repo_id, args.output_dir, args.force), indent=2))


if __name__ == "__main__":
    main()
