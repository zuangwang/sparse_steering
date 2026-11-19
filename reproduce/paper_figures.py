"""Redraw all seven paper data figures from bundled four-seed results (CPU only)."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from sparse_steering.analysis.paper_figures import render_all

    render_all(args.results, args.output)
    print("Wrote seven paper data figures to", args.output)


if __name__ == "__main__":
    main()
