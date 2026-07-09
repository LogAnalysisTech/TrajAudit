"""
Collect swe-agent reference trajectories for RootSE swe-agent instances
from the public S3 bucket: s3://scaleapi-results/swe-bench-pro/

For each of the 49 RootSE swe-agent instances × 6 backbones, downloads
the trajectory and eval result, determines pass/fail via fail_to_pass
tests, and saves under:

  reference_trajs/swe-agent/swe-bench-pro/{instance_id}/success/{backbone}.json
  reference_trajs/swe-agent/swe-bench-pro/{instance_id}/fail/{backbone}.json

Step format matches RootSE swe-agent exactly:
  {index (str), action (str), observation (str), response (str),
   thought (str), execution_time (float), state (dict), extra_info (dict)}

Usage:
  cd agent_system_log_normalization
  python baseline/FAMAS/collect_swe_agent_pro_trajs.py
"""

import json
import os
import subprocess
import tempfile

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOTSE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "RootSE", "swe-agent")
REF_BASE   = os.path.join(os.path.dirname(__file__), "reference_trajs", "swe-agent", "swe-bench-pro")
S3_BASE    = "s3://scaleapi-results/swe-bench-pro"

# backbone name → S3 folder
S3_FOLDERS = {
    "claude-45-sonnet": "claude-45sonnet-10132025",
    "gpt-5":            "gpt-5-250-turns-10132025",
    "gemini-2-5-pro":   "gemini-2-5-pro-preview-250-turns-debug-oct22",
    "claude-45-haiku":  "claude-45haiku-10222025",
    "gpt-5-codex":      "gpt-5-codex-debug-oct22",
    "glm-4p5":          "glm-4p5-10222025",
}

# ---------------------------------------------------------------------------
# Step 1: build fail_to_pass lookup from RootSE data
# ---------------------------------------------------------------------------

def build_fail_to_pass_lookup() -> dict:
    """instance_id → set of test names that must pass (fail_to_pass)"""
    lookup = {}
    for model_dir in os.listdir(ROOTSE_DIR):
        model_path = os.path.join(ROOTSE_DIR, model_dir)
        if not os.path.isdir(model_path):
            continue
        for fname in os.listdir(model_path):
            if not fname.endswith(".json"):
                continue
            data = json.load(open(os.path.join(model_path, fname), encoding="utf-8"))
            iid  = data["instance_id"]
            ftp  = {
                k for k, v in data.get("failed_tests", {}).items()
                if v.get("expected") == "fail_to_pass"
            }
            if iid not in lookup:
                lookup[iid] = ftp
    return lookup

# ---------------------------------------------------------------------------
# Step 2: S3 helpers
# ---------------------------------------------------------------------------

def s3_download(s3_path: str, local_path: str) -> bool:
    r = subprocess.run(
        ["aws", "s3", "cp", s3_path, local_path, "--no-sign-request"],
        capture_output=True, text=True
    )
    return r.returncode == 0


def determine_passed(eval_json_path: str, fail_to_pass_tests: set) -> bool:
    """True if all fail_to_pass tests show PASSED in eval output."""
    if not fail_to_pass_tests:
        return False  # 无法判断，保守归为 fail
    data   = json.load(open(eval_json_path, encoding="utf-8"))
    tests  = {t["name"]: t["status"] for t in data.get("tests", [])}
    return all(tests.get(t) == "PASSED" for t in fail_to_pass_tests)

# ---------------------------------------------------------------------------
# Step 3: convert S3 trajectory steps → RootSE swe-agent format
# ---------------------------------------------------------------------------

def convert_traj(traj_path: str) -> list:
    data  = json.load(open(traj_path, encoding="utf-8"))
    steps = data.get("trajectory", [])
    result = []
    for idx, s in enumerate(steps):
        result.append({
            "index":          str(idx),
            "action":         s.get("action", ""),
            "observation":    s.get("observation", ""),
            "response":       s.get("response", ""),
            "thought":        s.get("thought", ""),
            "execution_time": s.get("execution_time", 0.0),
            "state":          s.get("state", {}),
            "extra_info":     s.get("extra_info", {}),
        })
    return result

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Building fail_to_pass lookup from RootSE...")
    ftp_lookup = build_fail_to_pass_lookup()
    instance_ids = sorted(ftp_lookup.keys())
    print(f"  {len(instance_ids)} unique instances\n")

    total_saved = 0
    total_missing = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for backbone, s3_folder in S3_FOLDERS.items():
            print(f"=== {backbone} ({s3_folder}) ===")

            for iid in instance_ids:
                traj_s3  = f"{S3_BASE}/{s3_folder}/traj/{iid}/{iid}.traj"
                eval_s3  = f"{S3_BASE}/{s3_folder}/eval/{iid}/_output.json"
                traj_tmp = os.path.join(tmpdir, "traj.json")
                eval_tmp = os.path.join(tmpdir, "eval.json")

                # Download traj
                if not s3_download(traj_s3, traj_tmp):
                    print(f"  [MISS] {iid} — traj not found")
                    total_missing += 1
                    continue

                # Download eval
                if not s3_download(eval_s3, eval_tmp):
                    print(f"  [MISS] {iid} — eval not found")
                    total_missing += 1
                    continue

                # Determine pass/fail
                passed  = determine_passed(eval_tmp, ftp_lookup[iid])
                outcome = "success" if passed else "fail"

                # Convert steps
                steps = convert_traj(traj_tmp)

                # Save
                dest_dir  = os.path.join(REF_BASE, iid, outcome)
                os.makedirs(dest_dir, exist_ok=True)
                dest_file = os.path.join(dest_dir, f"{backbone}.json")
                with open(dest_file, "w", encoding="utf-8") as f:
                    json.dump({"passed": passed, "steps": steps}, f,
                              ensure_ascii=False, indent=2)

                status = "PASS" if passed else "FAIL"
                print(f"  [{status}] {iid[-50:]} → {outcome}/{backbone}.json ({len(steps)} steps)")
                total_saved += 1

            print()

    print(f"Done. Saved {total_saved}, missing {total_missing}.")


if __name__ == "__main__":
    main()
