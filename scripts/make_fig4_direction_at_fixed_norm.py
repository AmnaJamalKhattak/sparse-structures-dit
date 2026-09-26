"""Draw the alignment-with-v*-at-fixed-norm figure from the direction vs magnitude experiment.

    python scripts/make_fig4_direction_at_fixed_norm.py [--data results/q13] [--out figures]

Inputs (see ``results/q13/PROVENANCE.md``): the depth sweep and the rotation run of
``notebooks/direction_vs_magnitude.ipynb`` on FLUX.1-schnell. Output:
``fig_direction_at_fixed_norm.pdf`` and ``.png``.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ditsinks import paper_figures as P  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "results" / "q13")
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    depth = pd.read_csv(args.data / "q13_depth_response_flux1-schnell.csv")
    rotation = pd.read_csv(args.data / "q13_rotation_retention_flux1-schnell.csv")
    fig = P.fig_direction_at_fixed_norm(depth, rotation)
    args.out.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.out / f"fig_direction_at_fixed_norm.{suffix}", dpi=300)
    print("written:", args.out / "fig_direction_at_fixed_norm.pdf")


if __name__ == "__main__":
    main()
