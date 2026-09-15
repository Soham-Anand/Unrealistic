#!/usr/bin/env python3
"""Full-text audit of the exact StarCoder slices used in Phase 4 training.

Decodes the tokenized .bin slices back to plain text, saves the exact slice,
and scans EVERY document for license / copyright markers so we can quantify
(and point to) GPL / copyleft / copyrighted content in the actual training data.

Outputs (into OUT_DIR, one subdir per slice):
  <slice>.txt                 - the full decoded text (EOS-separated) = the exact slice
  <slice>_LICENSED.txt        - every doc containing any license marker
  <slice>_GPL.txt             - every doc with GPL/copyleft markers, with snippet
  <slice>_report.txt          - summary counts
  report_all.txt              - aggregate across slices
"""

import os, sys, re, glob, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_OUT = "/var/folders/mb/wkgc1cfj4xg2x4knbg0npcrr0000gn/T/opencode/starcoder_audit"

CHUNK = 1 << 28  # 256M uint16 at a time

# ---- License classifiers (applied to decoded document text) ----

# Strong copyleft / restrictive markers -> the "contamination" we care about
GPL_PAT = re.compile(
    r"GNU (?:GENERAL|AFFERO) PUBLIC LICENSE"
    r"|GNU Lesser General Public"
    r"|GNU GPL|GPL[- ]v?[23](\.\d)?"
    r"|Free Software Foundation"
    r"|this (?:program|software|library) is free software"
    r"|AGPL|LGPL"
    r"|GPL", re.I)

# Permissive markers (lower concern): MIT, Apache-2.0, BSD, ISC, MPL
PERM_PAT = re.compile(
    r"MIT[ -]?license|MIT[ -]?[Ll]icense"
    r"|Apache[- ](?:License )?2\.0|Apache[( ]2\.0|Licensed under the Apache"
    r"|BSD[- ]?\d-?[Cc]lause|BSD[- ]?[Ll]icense|Redistribution and use in source"
    r"|ISC[ -]?[Ll]icense"
    r"|Mozilla Public License|MPL[- ]2\.0", re.I)

# Any license / copyright mention at all
ANY_PAT = re.compile(
    r"licen[cs]e|copyright|©|copyleft|SPDX|free software", re.I)

# Explicit "all rights reserved" / proprietary
PROP_PAT = re.compile(r"All rights reserved|proprietary|not for redistribution", re.I)


def walk_docs(path, eos):
    """Stream docs (list of token ids) from a tokenized bin."""
    with open(path, "rb") as f:
        carry = np.zeros(0, dtype=np.uint16)
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            arr = np.concatenate([carry, np.frombuffer(chunk, dtype=np.uint16)])
            carry = np.zeros(0, dtype=np.uint16)
            start = 0
            for i, v in enumerate(arr):
                if v == eos:
                    yield arr[start:i].tolist()
                    start = i + 1
            if start < len(arr):
                carry = arr[start:]


def audit_slice(name, bin_path, tok, out_dir):
    sub = os.path.join(out_dir, name)
    os.makedirs(sub, exist_ok=True)
    eos = tok.eos_token

    txt_path = os.path.join(sub, f"{name}.txt")
    lic_path = os.path.join(sub, f"{name}_LICENSED.txt")
    gpl_path = os.path.join(sub, f"{name}_GPL.txt")
    rep_path = os.path.join(sub, f"{name}_report.txt")

    stats = {"docs": 0, "any": 0, "perm": 0, "gpl": 0, "prop": 0, "empty": 0}

    with open(txt_path, "w") as ftxt, \
         open(lic_path, "w") as flic, \
         open(gpl_path, "w") as fgpl:

        for di, ids in enumerate(walk_docs(bin_path, eos)):
            text = tok.decode(ids)
            stats["docs"] += 1
            if not text.strip():
                stats["empty"] += 1
                continue

            # write full text
            ftxt.write(f"\n===== DOC {di} =====\n{text}\n")

            has_any = bool(ANY_PAT.search(text))
            has_perm = bool(PERM_PAT.search(text))
            has_gpl = bool(GPL_PAT.search(text))
            has_prop = bool(PROP_PAT.search(text))
            if has_any:
                stats["any"] += 1
                flic.write(f"\n===== DOC {di} =====\n{text}\n")
            if has_perm:
                stats["perm"] += 1
            if has_gpl:
                stats["gpl"] += 1
                fgpl.write(f"\n===== DOC {di} =====\n{text}\n")
            if has_prop:
                stats["prop"] += 1

    with open(rep_path, "w") as f:
        f.write(f"SLICE: {name}\n")
        f.write(f"total docs : {stats['docs']}\n")
        f.write(f"empty      : {stats['empty']}\n")
        f.write(f"any license: {stats['any']} ({100*stats['any']/max(1,stats['docs']):.1f}%)\n")
        f.write(f"permissive : {stats['perm']} ({100*stats['perm']/max(1,stats['docs']):.1f}%)\n")
        f.write(f"GPL/copyleft: {stats['gpl']} ({100*stats['gpl']/max(1,stats['docs']):.1f}%)\n")
        f.write(f"proprietary: {stats['prop']} ({100*stats['prop']/max(1,stats['docs']):.1f}%)\n")

    print(f"[{name}] docs={stats['docs']} any={stats['any']} "
          f"perm={stats['perm']} GPL={stats['gpl']} prop={stats['prop']}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", nargs="*", default=None)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    from src.data.tokenizer import Tokenizer
    tok = Tokenizer("data/tokenizer/phase1.model")
    os.makedirs(args.out, exist_ok=True)

    SLICES = {
        "python": "data/phase4/python_raw.bin",
        "cpp": "data/phase4/cpp_raw.bin",
        "c": "data/phase4/c_raw.bin",
        "javascript": "data/phase4/javascript_raw.bin",
        "typescript": "data/phase4/typescript_raw.bin",
        "html": "data/phase4/html_raw.bin",
        "css": "data/phase4/css_raw.bin",
    }
    if args.slices:
        SLICES = {k: v for k, v in SLICES.items() if k in args.slices}
        if not SLICES:
            print("no matching slices")
            return

    agg = {}
    for name, path in SLICES.items():
        if not os.path.exists(path):
            print(f"SKIP {name}: {path} missing")
            continue
        s = audit_slice(name, path, tok, args.out)
        agg[name] = s

    with open(os.path.join(args.out, "report_all.txt"), "w") as f:
        f.write("AGGREGATE AUDIT - exact StarCoder slices in Phase 4 training\n")
        f.write(f"{'slice':12s} {'docs':>8s} {'any':>8s} {'perm':>8s} {'GPL':>8s} {'prop':>8s}\n")
        tot = {"any":0,"perm":0,"gpl":0,"prop":0,"docs":0}
        for name, s in agg.items():
            f.write(f"{name:12s} {s['docs']:8d} {s['any']:8d} {s['perm']:8d} {s['gpl']:8d} {s['prop']:8d}\n")
            for k in tot: tot[k]+=s[k]
        f.write("-"*60+"\n")
        f.write(f"{'TOTAL':12s} {tot['docs']:8d} {tot['any']:8d} {tot['perm']:8d} {tot['gpl']:8d} {tot['prop']:8d}\n")
    print("\nReport written to", os.path.join(args.out, "report_all.txt"))


if __name__ == "__main__":
    main()
