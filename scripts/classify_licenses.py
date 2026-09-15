#!/usr/bin/env python3
"""Precise per-doc license classification for a decoded text slice.

Unlike the broad scanner, this determines the DOMINANT license each document
actually declares (from its licence/SPDX/header text), so we can report how
much true GPL/copyleft vs permissive content is in the training slice.

Usage:
  python3 classify_licenses.py <slice_GPL.txt>   # classify every doc in the GPL file
  python3 classify_licenses.py --all <out_dir>   # classify all slices' GPL files
"""

import os, re, sys, glob, argparse


# Dominant-license classifiers, tried in order. Returns a license label or None.
def classify(text):
    low = text[:4000]
    l = low.lower()

    # --- copyleft first (most restrictive wins) ---
    if re.search(r"gnu affero general public license|agpl[- ]v?3?|gnu agpl", l):
        return "GPL-AGPL"
    if re.search(r"gnu lesser general public license|lgpl[- ]v?3?|gnu lgpl", l):
        return "GPL-LGPL"
    if re.search(
        r"gnu general public license as published by the free software foundation"
        r"|\bversions of the (gnu )?gpl\b as published by the free software foundation"
        r"|gpl[- ](?:v?)(?:2|3)(?:\.0)?(?: or later|\+)?\b"
        r"|spdx[- ](?:license[- ])?id:?\s*gpl[- ]?(?:v|2|3)",
        l
    ):
        # "GPL-2.0-or-later" style SPDX or "GNU GPL ... Free Software Foundation"
        return "GPL"

    # Eclipse (not GPL, but copyleft-ish): Eclipse Public License
    if re.search(r"eclipse public license|epl[- ]1\.0|epl[- ]2\.0", l):
        return "EPL(copyleft)"

    # Mozilla
    if re.search(r"mozilla public license|mpl[- ]2\.0", l):
        return "MPL(copyleft)"

    # --- permissive ---
    if re.search(r"apache[ -](?:license[ -]?)?2\.0|apache[ \(,;]2\.0|licensed under the apache|spdx[- ]license[- ]id:?\s*apache", l):
        return "Apache-2.0"
    if re.search(r"mit[ -]license|spdx[- ]license[- ]id:?\s*mit\b|permission is hereby granted, free of charge", l):
        return "MIT"
    if re.search(r"bsd[- ]\d[- ]clause|redistribution and use in source and binary forms", l):
        return "BSD"
    if re.search(r"isc[ -]license|spdx[- ]license[- ]id:?\s*isc\b", l):
        return "ISC"

    # GPL mentioned but no confident declaration -> generic
    if re.search(r"\bgpl\b|free software foundation", l):
        return "GPL?(loose)"

    return None


def classify_docs(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    docs = [d for d in text.split("\n===== DOC ") if d.strip()]
    from collections import Counter
    counts = Counter()
    examples = {}
    for d in docs:
        # strip the leading "NN ===== " to get content
        content = d.split(" ===== ", 1)[-1] if " ===== " in d else d
        lab = classify(content)
        if lab:
            counts[lab] += 1
            examples.setdefault(lab, content[:220].replace("\n", " | "))
    return counts, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", default=None, help="audit out_dir containing <slice>/<slice>_GPL.txt")
    ap.add_argument("paths", nargs="*")
    args = ap.parse_args()

    paths = list(args.paths)
    if args.all:
        paths = sorted(glob.glob(os.path.join(args.all, "*", "*_GPL.txt")))

    if not paths:
        print("no paths given; pass slice_GPL.txt files or use --all <out_dir>")
        return

    for p in paths:
        if not os.path.exists(p):
            print(f"MISSING {p}")
            continue
        counts, examples = classify_docs(p)
        print("=" * 70)
        print(f"FILE: {p}")
        for lab, n in counts.most_common():
            print(f"  {lab:14s} {n:5d}")
        print("  examples:")
        for lab, ex in list(examples.items())[:2]:
            print(f"    [{lab}] {ex}")


if __name__ == "__main__":
    main()
