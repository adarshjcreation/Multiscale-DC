"""Command line: downward continuation of a gravity grid with the pretrained multiscale networks.

    python -m multiscale_dc examples/data/example_obs_h100m.csv --spacing 50 --height 100 --plot result.png
"""
import argparse
import os
import time

from . import DEFAULT_PASSES, ROUTINES, WEIGHTS, downward_continue, load_grid, plot_result, save_grid


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m multiscale_dc",
        description="Downward continuation of a gridded gravity anomaly with the pretrained multiscale networks.",
        epilog="The grid must be regular (same spacing in x and y), at least 32 x 32 nodes and without gaps. "
               "The result has the unit of the input and is written to OUTPUT (default: <input>_dc.npy).")
    p.add_argument("input", help="grid file: .npy, .csv/.txt (2-D table of values) or .mat")
    p.add_argument("--spacing", type=float, required=True, help="grid spacing, e.g. in m")
    p.add_argument("--height", type=float, required=True,
                   help="continuation distance in the unit of --spacing (at most 2 x spacing)")
    p.add_argument("-o", "--output", help="output file, .npy or .csv")
    p.add_argument("--plot", metavar="PNG", help="also save a figure of the input and the result")
    p.add_argument("--unit", default="mGal", help="unit shown in the figure (default: mGal)")
    p.add_argument("--var", help=".mat input: name of the variable holding the grid")
    p.add_argument("--model", choices=sorted(DEFAULT_PASSES), default="multiscale_unet",
                   help="network (default: multiscale_unet)")
    p.add_argument("--weights", choices=sorted(WEIGHTS), default="field",
                   help="weight set (default: field, trained on positive and negative anomalies)")
    p.add_argument("--routine", choices=ROUTINES, default="stabilized", help="multiscale routine (default: stabilized)")
    p.add_argument("--passes", type=int, help="scaling passes (default: 2 for the U-Net, 5 for the DiT)")
    p.add_argument("--stride", type=int, default=8, help="step between the 32 x 32 tiles in nodes (default: 8)")
    p.add_argument("--pad", type=int, default=16,
                   help="extend the grid by this many nodes against edge effects, 0 = off (default: 16)")
    p.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    args = p.parse_args(argv)

    grid = load_grid(args.input, args.var)
    print(f"input  {args.input}: {grid.shape[0]} x {grid.shape[1]} nodes, spacing {args.spacing:g}, "
          f"values {grid.min():.4g} to {grid.max():.4g}")
    t0 = time.time()
    out = downward_continue(grid, args.spacing, args.height, model=args.model, weights=args.weights,
                            routine=args.routine, passes=args.passes, stride=args.stride, pad=args.pad,
                            device=args.device)
    path = args.output or os.path.splitext(args.input)[0] + "_dc.npy"
    save_grid(path, out)
    print(f"output {path}: continued {args.height:g} downward with {args.model} ({args.weights} weights) in "
          f"{time.time() - t0:.1f} s, values {out.min():.4g} to {out.max():.4g}")
    if args.plot:
        plot_result(grid, out, args.spacing, args.height, args.plot, unit=args.unit)
        print(f"figure {args.plot}")


if __name__ == "__main__":
    main()
