"""Consistent command-line interfaces for dataset preparation, training, and export."""

import argparse
import importlib
from pathlib import Path
import subprocess
import sys

from shared.paths import DATASETS_DIR, OUTPUTS_DIR, REPO_ROOT


def run_classifier(dataset, argv=None):
    """Dispatch a dataset's train or export command to the shared implementation."""
    parser = argparse.ArgumentParser(
        description=f"Train or export probabilities for {dataset}.", allow_abbrev=False
    )
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser(
        "train", help="Train the classifier", allow_abbrev=False
    )
    export = commands.add_parser(
        "export", help="Export frozen classifier probabilities", allow_abbrev=False
    )
    for command in (train, export):
        command.add_argument(
            "--config",
            type=Path,
            default=REPO_ROOT / "configs" / f"train_{dataset}.yaml",
            help="Training YAML (defaults to this dataset's config; relative paths use the repo root)",
        )
    train.add_argument(
        "--resume", action="store_true", help="Resume from the saved checkpoint"
    )
    export.add_argument(
        "--checkpoint",
        type=Path,
        help="Fine-tuned weights (default: checkpoints/<dataset>/best.pt)",
    )
    args = parser.parse_args(argv)
    script = (
        "train_classifier.py" if args.command == "train" else "export_probabilities.py"
    )
    command = [
        sys.executable,
        str(Path(__file__).with_name(script)),
        "--config",
        str(args.config.expanduser()),
    ]
    if args.command == "train" and args.resume:
        command.append("--resume")
    if args.command == "export" and args.checkpoint is not None:
        command.extend(["--checkpoint", str(args.checkpoint.expanduser())])
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None


def manifest_arguments(dataset, description, argv=None):
    """Parse common manifest options without reading data or creating directories."""
    parser = argparse.ArgumentParser(description=description, allow_abbrev=False)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATASETS_DIR,
        help="Dataset root containing COVID/, ISIC_2019/, EyePACS/, and APTOS_2019/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_DIR / dataset / "manifest.csv",
        help="Destination CSV (default: outputs/<dataset>/manifest.csv)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing manifest"
    )
    args = parser.parse_args(argv)
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.output.exists() and not args.overwrite:
        parser.error(
            f"Output already exists: {args.output}; use --overwrite to replace it"
        )
    return args


def save_manifest(manifest, args):
    """Write the manifest and print split counts; preserve existing files by default."""
    if "path" in manifest.columns:
        manifest = manifest.copy()
        root = Path(args.data_dir).resolve()

        def portable(value):
            path = Path(value)
            try:
                return (Path("data") / path.relative_to(root)).as_posix()
            except ValueError:
                return str(path)

        manifest["path"] = manifest["path"].map(portable)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False, mode="w" if args.overwrite else "x")
    print(manifest.groupby(["split", "label"]).size().unstack(fill_value=0))
    print("total", manifest.groupby("split").size().to_dict())


def ensure_manifest(config):
    """Prepare fixed dataset splits on first use, never replace an existing manifest."""
    output = Path(config["manifest"])
    if output.is_file():
        return output
    dataset = config["experiment"]
    if dataset not in {"covid19", "isic2019", "eyepacs_to_aptos"}:
        raise FileNotFoundError(f"Prepare the custom manifest before running: {output}")
    module = importlib.import_module(f"experiments.{dataset}.prepare_manifest")
    print(f"Preparing {dataset} manifest from {DATASETS_DIR}", flush=True)
    manifest = module.build_manifest(DATASETS_DIR)
    save_manifest(manifest, argparse.Namespace(
        data_dir=DATASETS_DIR, output=output, overwrite=False,
    ))
    return output
