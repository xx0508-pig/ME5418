"""Prepare pinned LIBERO assets and project-local configuration (Linux setup).

Uses the standard library so assets can be downloaded during dependency install.
Run through scripts/local.sh after creating .venv. Does not import LIBERO, whose
first import would otherwise ask an interactive dataset-path question.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / ".local/libero-assets"
REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
API = "https://huggingface.co/api/datasets/lerobot/libero-assets"
DOWNLOAD = f"https://huggingface.co/datasets/lerobot/libero-assets/resolve/{REVISION}"


def valid_file(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    content = path.read_bytes()
    if "lfs" in entry:
        return hashlib.sha256(content).hexdigest() == entry["lfs"]["oid"]
    blob = f"blob {len(content)}\0".encode() + content
    return hashlib.sha1(blob).hexdigest() == entry["oid"]


def download_one(entry):
    path = (ASSETS / entry["path"]).resolve()
    if not path.is_relative_to(ASSETS.resolve()):
        raise ValueError(f"Unexpected asset path: {entry['path']}")
    if valid_file(path, entry):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".download")
    for attempt in range(3):
        try:
            with urlopen(f"{DOWNLOAD}/{entry['path']}", timeout=90) as response:
                with temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            if not valid_file(temporary, entry):
                raise ValueError(f"Asset checksum mismatch: {entry['path']}")
            temporary.replace(path)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-only", action="store_true")
    args = parser.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / ".local/libero-assets-manifest.json"
    if manifest.exists():
        metadata = json.loads(manifest.read_text())
        if metadata["revision"] != REVISION:
            raise ValueError("Existing asset manifest uses another revision")
        entries = metadata["files"]
    else:
        with urlopen(f"{API}/tree/{REVISION}?recursive=true&limit=1000", timeout=60) as response:
            if response.headers.get("Link"):
                raise RuntimeError("Asset listing is paginated; do not use an incomplete manifest")
            entries = [entry for entry in json.load(response) if entry["type"] == "file"]
        manifest.write_text(json.dumps({"revision": REVISION, "files": entries}, indent=2) + "\n")
    with ThreadPoolExecutor(max_workers=8) as pool:
        for count, _ in enumerate(pool.map(download_one, entries), 1):
            if count % 50 == 0 or count == len(entries):
                print(f"Verified assets: {count}/{len(entries)}", flush=True)
    if args.assets_only:
        return
    package_root = Path(distribution("hf-libero").locate_file("libero/libero")).resolve()
    asset_link = package_root / "assets"
    if not asset_link.exists() and not asset_link.is_symlink():
        asset_link.symlink_to(ASSETS, target_is_directory=True)
    elif asset_link.resolve() != ASSETS.resolve():
        raise RuntimeError(f"Existing assets were not replaced: {asset_link}")
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", ROOT / ".local/libero-config"))
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.yaml"
    if not config_file.exists():
        config = {"benchmark_root": str(package_root),
                  "bddl_files": str(package_root / "bddl_files"),
                  "init_states": str(package_root / "init_files"),
                  "datasets": str(ROOT / ".local/datasets"),
                  "assets": str(ASSETS)}
        # JSON is also valid YAML; avoid importing any robotics package here.
        config_file.write_text(json.dumps(config, indent=2) + "\n")
    print(f"LIBERO assets: {ASSETS}\nLIBERO config: {config_file}")


if __name__ == "__main__":
    main()
