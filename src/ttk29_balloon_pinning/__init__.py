"""Project-level CLI for the balloon, PINN, CoSTA, and sparse-sensor workflows."""

from __future__ import annotations

import argparse
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    """Construct the main project CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the Balloon-Windkessel, PINN, CoSTA, and sparse-sensor analysis workflows "
            "with sensible defaults and a single project entry point."
        )
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = False

    for command_name in ("physical", "pinn", "costa", "compare"):
        subparser = subparsers.add_parser(command_name, help=f"Run the {command_name} model pipeline")
        subparser.add_argument("--data", type=Path, help="CSV with time,input,bold")
        subparser.add_argument(
            "--subject",
            help="BIDS subject ID; defaults to sub-10159 when no other input is provided",
        )
        subparser.add_argument("--data-root", type=Path, default=Path("data/ds000030"))
        subparser.add_argument("--bold-path", type=Path, help="Preprocessed BOLD NIfTI; overrides the raw subject path")
        subparser.add_argument("--events-path", type=Path, help="Events TSV; defaults to the subject's raw BIDS events file")
        subparser.add_argument(
            "--roi-mask",
            type=Path,
            help="Aligned ROI mask NIfTI; if omitted, extract the EPI-masked whole-brain mean",
        )
        subparser.add_argument(
            "--synthetic",
            action="store_true",
            help="Use deterministic synthetic input instead of real subject or CSV data",
        )
        subparser.add_argument("--output-dir", type=Path, help="Directory for results; defaults to a timestamped folder under results/")
        subparser.add_argument("--epochs", type=int, default=2000)
        subparser.add_argument("--no-fit-physical", action="store_true", help="Use default Balloon parameters")

    sensor_parser = subparsers.add_parser("sensor", help="Run the sparse sensor-placement analysis")
    sensor_parser.add_argument("--data-root", type=Path, default=Path("data/ds000030"))
    sensor_parser.add_argument("--task", default="rest")
    sensor_parser.add_argument("--max-participants", type=int, default=20)
    sensor_parser.add_argument("--include-all-diagnoses", action="store_true")
    sensor_parser.add_argument("--max-mean-fd", type=float, default=0.5)
    sensor_parser.add_argument("--voxel-size-mm", type=float, default=4.0)
    sensor_parser.add_argument("--smoothing-fwhm-mm", type=float, default=6.0)
    sensor_parser.add_argument("--global-signal", action="store_true")
    sensor_parser.add_argument("--cache-dir", type=Path, default=Path("data/cleaned"))
    sensor_parser.add_argument("--k-values")
    sensor_parser.add_argument("--k-fraction", type=float, default=0.05)
    sensor_parser.add_argument("--k-points", type=int, default=8)
    sensor_parser.add_argument("--rank", type=int)
    sensor_parser.add_argument("--n-random-draws", type=int, default=10)
    sensor_parser.add_argument("--n-folds", type=int, default=5)
    sensor_parser.add_argument("--include-l1", action="store_true")
    sensor_parser.add_argument("--l1-k-values")
    sensor_parser.add_argument("--l1-alpha", type=float)
    sensor_parser.add_argument("--l1-iterations", type=int, default=40)
    sensor_parser.add_argument("--l1-test-timepoints", type=int, default=40)
    sensor_parser.add_argument("--output-dir", type=Path, default=Path("results/sensor_placement"))
    sensor_parser.add_argument("--seed", type=int, default=0)

    summary_parser = subparsers.add_parser("summary", help="Summarize the newest saved results")
    summary_parser.add_argument("path", type=Path, nargs="?", help="Run directory or .npz file; defaults to the newest run under results/")
    summary_parser.add_argument("--plot", action="store_true", help="Plot the comparison summary for the resolved run")
    summary_parser.add_argument("--latest", action="store_true", help="Explicitly resolve the newest run under results/")

    latest_parser = subparsers.add_parser("latest", help="Show the newest saved results summary")
    latest_parser.add_argument("--plot", action="store_true", help="Also save a comparison plot for the newest run")

    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the repository-level CLI with a minimal, discoverable interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command in {"physical", "pinn", "costa", "compare"}:
        from model_comparison.run import main as comparison_main

        command_argv = ["--mode", args.command]
        if getattr(args, "data", None) is not None:
            command_argv.extend(["--data", str(args.data)])
        if getattr(args, "subject", None) is not None:
            command_argv.extend(["--subject", args.subject])
        if getattr(args, "data_root", None) is not None:
            command_argv.extend(["--data-root", str(args.data_root)])
        if getattr(args, "bold_path", None) is not None:
            command_argv.extend(["--bold-path", str(args.bold_path)])
        if getattr(args, "events_path", None) is not None:
            command_argv.extend(["--events-path", str(args.events_path)])
        if getattr(args, "roi_mask", None) is not None:
            command_argv.extend(["--roi-mask", str(args.roi_mask)])
        if getattr(args, "output_dir", None) is not None:
            command_argv.extend(["--output-dir", str(args.output_dir)])
        if getattr(args, "epochs", None) is not None:
            command_argv.extend(["--epochs", str(args.epochs)])
        if getattr(args, "no_fit_physical", False):
            command_argv.append("--no-fit-physical")
        if args.synthetic:
            command_argv.append("--synthetic")
        comparison_main(command_argv)
        return

    if args.command == "sensor":
        from sensor_placement.run import main as sensor_main

        sensor_argv = []
        for option, value in (
            ("--data-root", args.data_root),
            ("--task", args.task),
            ("--max-participants", args.max_participants),
            ("--max-mean-fd", args.max_mean_fd),
            ("--voxel-size-mm", args.voxel_size_mm),
            ("--smoothing-fwhm-mm", args.smoothing_fwhm_mm),
            ("--cache-dir", args.cache_dir),
            ("--k-fraction", args.k_fraction),
            ("--k-points", args.k_points),
            ("--n-random-draws", args.n_random_draws),
            ("--n-folds", args.n_folds),
            ("--l1-iterations", args.l1_iterations),
            ("--l1-test-timepoints", args.l1_test_timepoints),
            ("--output-dir", args.output_dir),
            ("--seed", args.seed),
        ):
            sensor_argv.extend([option, str(value)])
        for option, value in (
            ("--k-values", args.k_values),
            ("--rank", args.rank),
            ("--l1-k-values", args.l1_k_values),
            ("--l1-alpha", args.l1_alpha),
        ):
            if value is not None:
                sensor_argv.extend([option, str(value)])
        if args.include_all_diagnoses:
            sensor_argv.append("--include-all-diagnoses")
        if args.global_signal:
            sensor_argv.append("--global-signal")
        if args.include_l1:
            sensor_argv.append("--include-l1")
        sensor_main(sensor_argv)
        return

    if args.command in {"summary", "latest"}:
        from read_results import main as read_main

        if args.command == "latest":
            read_main(["--latest"] + (["--plot"] if args.plot else []))
            return
        read_main(([str(args.path)] if args.path is not None else []) + (["--plot"] if args.plot else []) + (["--latest"] if args.latest else []))
        return

    parser.print_help()
