"""
Build the selected_refs/ folder by applying the MAX_REF_TRAJS selection
mechanism to every instance in reference_trajs/.

Output layout (mirrors reference_trajs/):
  baseline/FAMAS/selected_refs/{agent}/{benchmark}/{instance_id}/success/*.json
  baseline/FAMAS/selected_refs/{agent}/{benchmark}/{instance_id}/fail/*.json

Each output JSON keeps the original file content unchanged; only the set of
files written is reduced to the selected subset.

Usage:
  cd agent_system_log_normalization
  python baseline/FAMAS/build_selected_refs.py

Re-running is idempotent (overwrites existing selected_refs/).
"""

import json
import math
import os
import shutil

# ── config ────────────────────────────────────────────────────────────────────
MAX_REF_TRAJS = 5

REF_BASE = os.path.join(os.path.dirname(__file__), "reference_trajs")
OUT_BASE = os.path.join(os.path.dirname(__file__), "selected_refs")
# ──────────────────────────────────────────────────────────────────────────────


def select_files(pass_files: list, fail_files: list, max_n: int):
    """
    Select up to max_n files with balanced pass/fail ratio.

    Longer trajectories (more steps) are preferred within each group.
    Returns (selected_pass_files, selected_fail_files).
    """
    def sort_key(fpath):
        try:
            data = json.load(open(fpath, encoding="utf-8"))
            steps = data.get("steps", data.get("traj", data.get("original_traj", [])))
            return len(steps)
        except Exception:
            return 0

    # Sort each group: longest first
    pass_files = sorted(pass_files, key=sort_key, reverse=True)
    fail_files = sorted(fail_files, key=sort_key, reverse=True)

    total = len(pass_files) + len(fail_files)
    if total <= max_n:
        return pass_files, fail_files   # no selection needed

    n_pass_avail = len(pass_files)
    n_fail_avail = len(fail_files)

    # Balanced allocation: target ceil(max_n/2) from each group
    n_minority_target = math.ceil(max_n / 2)
    n_pass_take = min(n_pass_avail, n_minority_target)
    n_fail_take = min(n_fail_avail, max_n - n_pass_take)
    n_pass_take = min(n_pass_avail, max_n - n_fail_take)  # fill leftover

    return pass_files[:n_pass_take], fail_files[:n_fail_take]


def main():
    if not os.path.isdir(REF_BASE):
        print(f"reference_trajs/ not found at {REF_BASE}")
        return

    total_instances = 0
    total_selected  = 0
    total_original  = 0

    for agent in sorted(os.listdir(REF_BASE)):
        agent_ref = os.path.join(REF_BASE, agent)
        if not os.path.isdir(agent_ref):
            continue

        for bench in sorted(os.listdir(agent_ref)):
            bench_ref = os.path.join(agent_ref, bench)
            if not os.path.isdir(bench_ref):
                continue

            for inst in sorted(os.listdir(bench_ref)):
                inst_ref = os.path.join(bench_ref, inst)
                if not os.path.isdir(inst_ref):
                    continue

                # Collect source files
                success_dir = os.path.join(inst_ref, "success")
                fail_dir    = os.path.join(inst_ref, "fail")

                pass_files = []
                fail_files = []

                if os.path.isdir(success_dir):
                    pass_files = [
                        os.path.join(success_dir, f)
                        for f in sorted(os.listdir(success_dir))
                        if f.endswith(".json")
                    ]
                if os.path.isdir(fail_dir):
                    fail_files = [
                        os.path.join(fail_dir, f)
                        for f in sorted(os.listdir(fail_dir))
                        if f.endswith(".json")
                    ]

                # Legacy layout: JSON files directly under inst_ref
                if not pass_files and not fail_files:
                    for fname in sorted(os.listdir(inst_ref)):
                        if not fname.endswith(".json"):
                            continue
                        fpath = os.path.join(inst_ref, fname)
                        try:
                            data = json.load(open(fpath, encoding="utf-8"))
                            passed = bool(data.get("passed", data.get("success", False)))
                            if passed:
                                pass_files.append(fpath)
                            else:
                                fail_files.append(fpath)
                        except Exception:
                            pass

                n_orig = len(pass_files) + len(fail_files)
                if n_orig == 0:
                    continue

                # Apply selection
                sel_pass, sel_fail = select_files(pass_files, fail_files, MAX_REF_TRAJS)
                n_sel = len(sel_pass) + len(sel_fail)

                # Write to selected_refs/
                inst_out = os.path.join(OUT_BASE, agent, bench, inst)
                out_success = os.path.join(inst_out, "success")
                out_fail    = os.path.join(inst_out, "fail")
                os.makedirs(out_success, exist_ok=True)
                os.makedirs(out_fail,    exist_ok=True)

                for src in sel_pass:
                    shutil.copy2(src, os.path.join(out_success, os.path.basename(src)))
                for src in sel_fail:
                    shutil.copy2(src, os.path.join(out_fail,    os.path.basename(src)))

                status = "" if n_sel == n_orig else f"  (trimmed {n_orig} → {n_sel})"
                print(f"  {agent}/{bench}/{inst}"
                      f"  pass={len(sel_pass)}  fail={len(sel_fail)}{status}")

                total_instances += 1
                total_selected  += n_sel
                total_original  += n_orig

    print(f"\nDone. {total_instances} instances, "
          f"{total_selected}/{total_original} trajectories selected "
          f"(MAX_REF_TRAJS={MAX_REF_TRAJS}).")
    print(f"Output: {OUT_BASE}")


if __name__ == "__main__":
    main()
