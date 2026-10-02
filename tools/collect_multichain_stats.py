#!/usr/bin/env python
"""
Collect multichain diagnostics summaries from multiple Phase 2 runs.

Looks for traces/*_phase2_multichain_diagnostics.json under each run dir
and prints a compact summary.

Usage:
  python tools/collect_multichain_stats.py --run-dirs runA runB ...
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main(run_dirs):
    for rd in run_dirs:
        diag = Path(rd) / "traces"
        jsons = list(diag.glob("*_phase2_multichain_diagnostics.json"))
        if not jsons:
            print(f"{rd}: no multichain diagnostics found")
            continue
        j = json.loads(jsons[0].read_text())
        prefix = j.get("prefix", jsons[0].stem)
        chains = j.get("chains")
        draws_per_chain = j.get("draws_per_chain") or j.get("iterations_per_chain")
        max_rhat_tau = max(j.get("rhat_tau", [float('nan')]))
        max_rhat_mu = max(j.get("rhat_mu_tilde", [float('nan')]))
        max_rhat_r = max(j.get("rhat_r", [float('nan')]))
        print(f"{prefix}: chains={chains} draws/chain={draws_per_chain} "
              f"max_rhat_mu={max_rhat_mu:.3f} max_rhat_tau={max_rhat_tau:.3f} max_rhat_r={max_rhat_r:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    args = ap.parse_args()
    main(args.run_dirs)
