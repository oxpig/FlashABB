"""Command-line interface to FlashABB, a wrapper around the Python API
(`flash_abb.pretrained`, `pretrained_tap` and `pretrained_sss`).

Input is a CSV file with `heavy` and `light` columns and an optional `id`
column.  The modes are:

- `predict`: model each antibody's structure, written as `<id>.pdb` in the
  output directory.
- `tap`: predict the four FlashTAP developability scores (PSH, PPC, PNC
  and SFvCSP) and the probability that each is flagged, written as CSV.
- `embed`: compute FlashABB-SSS structure-aware per-residue embeddings,
  written as a NumPy `.npz` archive with one array per antibody, keyed
  by id.

The weights are bundled with FlashABB.
"""

import argparse
import sys

import numpy as np
import pandas as pd

MODES = ["predict", "tap", "embed"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="flashabb",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("input", help="CSV file with `heavy` and `light` columns")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="output directory for `predict`; CSV file for `tap`; "
        ".npz file for `embed`",
    )
    parser.add_argument(
        "--no-dlc",
        action="store_true",
        help="`predict` without the Dropout-LayerNorm Correction",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--device", default="cpu", help="e.g. `cpu` or `cuda`")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    table = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    missing = {"heavy", "light"} - set(table.columns)
    if missing:
        sys.exit(f"flashabb: input lacks column(s): {', '.join(sorted(missing))}")
    ids = list(table["id"] if "id" in table.columns else table.index.astype(str))
    sequences = [f"{h}|{l}" for h, l in zip(table["heavy"], table["light"])]

    # Imported late: loading torch is slow, and `--help` shouldn't wait.
    import torch
    import flash_abb

    if args.mode == "predict":
        model = flash_abb.pretrained(
            device=args.device, dlc_correction=not args.no_dlc
        )
        # `pretrained` takes a batch size but ignores it, so batch here.
        for start in range(0, len(sequences), args.batch_size):
            end = start + args.batch_size
            with torch.no_grad():
                result = model(sequences[start:end])
            result.to_pdbs(ids[start:end], pdb_dir=args.output)

    elif args.mode == "tap":
        model = flash_abb.pretrained_tap(device=torch.device(args.device))
        result = model(sequences, batch_size=args.batch_size)
        frame = pd.DataFrame(result.scores)
        if result.flag_probs is not None:
            flags = pd.DataFrame(result.flag_probs).add_suffix("_flag_prob")
            frame = pd.concat([frame, flags], axis=1)
            frame["any_flag_prob"] = result.any_flag_prob
        frame.insert(0, "id", ids)
        frame.to_csv(args.output, index=False)

    else:
        model = flash_abb.pretrained_sss(device=torch.device(args.device))
        result = model(sequences, batch_size=args.batch_size)
        embeddings = result.embeddings.cpu().numpy()
        mask = result.mask.cpu().numpy().astype(bool)
        np.savez(
            args.output,
            **{str(i): e[m] for i, e, m in zip(ids, embeddings, mask)},
        )


if __name__ == "__main__":
    main()
