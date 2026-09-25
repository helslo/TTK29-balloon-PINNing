"""Command-line entry point for the balloon model comparison."""


def main() -> None:
    """Run the repository-level comparison command."""
    from compare_models import main as comparison_main

    comparison_main()
