"""
Collect swe-agent-live reference trajectories from HuggingFace livesweagent org.

Downloads trajectories for each RootSE swe-agent-live instance from 5 backbone
datasets (those with confirmed pass/fail labels), converts them to the FAMAS
reference format, and places them under:

  reference_trajs/swe-agent-live/swe-bench/{instance_id}/success/{backbone}.json
  reference_trajs/swe-agent-live/swe-bench/{instance_id}/fail/{backbone}.json

Usage:
  cd agent_system_log_normalization
  python baseline/FAMAS/collect_swe_agent_live_trajs.py
"""

import json
import os
import re
import sys

from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# RootSE instance IDs for swe-agent-live (without 'instance_' prefix for HF lookup)
OUR_INSTANCES = [
    "django__django-15022",
    "django__django-15128",
    "django__django-15280",
    "matplotlib__matplotlib-20826",
    "matplotlib__matplotlib-25775",
    "pytest-dev__pytest-5787",
    "sphinx-doc__sphinx-10435",
    "sympy__sympy-13031",
    "sympy__sympy-13878",
]

# Backbone datasets with confirmed pass/fail labels
# Format: (HF repo suffix, backbone label, how to get resolved_ids)
DATASETS = [
    {
        "repo":    "claude-sonnet-4-5_swebench_verified_traj",
        "backbone": "claude-sonnet-4-5",
        "eval_file": "eval_result.json",
        "resolved_key": "resolved_ids",
    },
    {
        "repo":    "gpt-5_swebench_verified_traj",
        "backbone": "gpt-5",
        "eval_file": "eval_result.json",
        "resolved_key": "resolved_ids",
    },
    {
        "repo":    "gpt-5-mini_swebench_verified_traj",
        "backbone": "gpt-5-mini",
        "eval_file": "eval_result.json",
        "resolved_key": "resolved_ids",
    },
    {
        "repo":    "gemini_3_pro_swebench_verified_traj",
        "backbone": "gemini-3-pro",
        "eval_file": "result.json",
        "resolved_key": "resolved_ids",
    },
    {
        "repo":    "claude-opus-4-5_swebench_verified_traj",
        "backbone": "claude-opus-4-5",
        # No eval_result.json — labels fetched from swe-bench/experiments GitHub
        "eval_file": None,
        "resolved_ids_override": {
            "django__django-15022":       True,
            "django__django-15128":       True,
            "django__django-15280":       True,
            "matplotlib__matplotlib-20826": True,
            "matplotlib__matplotlib-25775": False,
            "pytest-dev__pytest-5787":    True,
            "sphinx-doc__sphinx-10435":   False,
            "sympy__sympy-13031":         True,
            "sympy__sympy-13878":         True,
        },
    },
]

# Folder layout
REF_BASE = os.path.join(
    os.path.dirname(__file__),
    "reference_trajs", "swe-agent-live", "swe-bench"
)

# ---------------------------------------------------------------------------
# Trajectory conversion
# ---------------------------------------------------------------------------

_BASH_BLOCK_RE = re.compile(r"```bash\s*(.*?)```", re.DOTALL)


def _extract_action(content: str) -> str:
    """Extract the bash command from an assistant message's ```bash block."""
    m = _BASH_BLOCK_RE.search(content)
    if m:
        return m.group(1).strip()
    return content.strip()


def _extract_thought(content: str) -> str:
    """Extract the THOUGHT section (text before the first ```bash block)."""
    parts = re.split(r"```bash", content, maxsplit=1)
    if len(parts) > 1:
        return parts[0].replace("THOUGHT:", "").strip()
    return ""


def convert_messages_to_steps(messages: list) -> list:
    """
    Convert livesweagent message list to step dicts matching the RootSE
    swe-agent-live format exactly:

      {
        "index":       "0",          # sequential string index
        "thought":     "...",        # THOUGHT text from assistant message
        "action":      "ls -la",     # bash command extracted from ```bash block
        "observation": "<returncode>0</returncode>\\n<output>\\n...\\n</output>",
        "response":    "THOUGHT: ...\n```bash\nls -la\n```"  # full assistant msg
      }

    Message structure (livesweagent):
      [0] system
      [1] user  — problem description (skipped)
      [2] assistant — THOUGHT + ```bash command```
      [3] user  — <returncode>…</returncode>\\n<output>…</output>
      [4] assistant — ...
      ...
    """
    steps = []
    step_index = 0
    i = 2  # skip system + initial problem message

    while i < len(messages):
        msg = messages[i]
        if msg["role"] == "assistant":
            response = msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
            thought = _extract_thought(response)
            action  = _extract_action(response)

            # Next user message = observation (keep raw <returncode>/<output> tags)
            observation = ""
            if i + 1 < len(messages) and messages[i + 1]["role"] == "user":
                obs_raw = messages[i + 1]["content"]
                observation = obs_raw if isinstance(obs_raw, str) else str(obs_raw)
                i += 2
            else:
                i += 1

            steps.append({
                "index":       str(step_index),
                "thought":     thought,
                "action":      action,
                "observation": observation,
                "response":    "",
            })
            step_index += 1
        else:
            i += 1

    return steps


# ---------------------------------------------------------------------------
# Main download logic
# ---------------------------------------------------------------------------

def get_resolved_ids(ds_cfg: dict) -> set:
    """Return the set of resolved instance IDs for a dataset config."""
    if "resolved_ids_override" in ds_cfg:
        return {k for k, v in ds_cfg["resolved_ids_override"].items() if v}

    eval_file = ds_cfg["eval_file"]
    if not eval_file:
        return set()

    repo_id = f"livesweagent/{ds_cfg['repo']}"
    path = hf_hub_download(repo_id, eval_file, repo_type="dataset")
    data = json.load(open(path))
    return set(data.get(ds_cfg["resolved_key"], []))


def download_and_save(ds_cfg: dict, resolved_ids: set):
    repo_id = f"livesweagent/{ds_cfg['repo']}"
    backbone = ds_cfg["backbone"]

    for instance_id in OUR_INSTANCES:
        fname = f"{instance_id}/{instance_id}.traj.json"
        try:
            local_path = hf_hub_download(repo_id, fname, repo_type="dataset")
        except Exception as e:
            print(f"  [{backbone}] {instance_id}: SKIP (not found: {e})")
            continue

        with open(local_path, encoding="utf-8") as f:
            traj_data = json.load(f)

        steps = convert_messages_to_steps(traj_data.get("messages", []))
        passed = instance_id in resolved_ids

        # Destination: reference_trajs/swe-agent-live/swe-bench/instance_{id}/success|fail/{backbone}.json
        folder_name = f"instance_{instance_id}"
        outcome_dir = "success" if passed else "fail"
        dest_dir = os.path.join(REF_BASE, folder_name, outcome_dir)
        os.makedirs(dest_dir, exist_ok=True)

        dest_file = os.path.join(dest_dir, f"{backbone}.json")
        with open(dest_file, "w", encoding="utf-8") as f:
            json.dump({"passed": passed, "steps": steps}, f, ensure_ascii=False, indent=2)

        status = "PASS" if passed else "FAIL"
        print(f"  [{backbone}] {instance_id}: {status} → {outcome_dir}/{backbone}.json  ({len(steps)} steps)")


def main():
    print(f"Output base: {REF_BASE}\n")

    for ds_cfg in DATASETS:
        backbone = ds_cfg["backbone"]
        print(f"=== {backbone} ({ds_cfg['repo']}) ===")
        try:
            resolved_ids = get_resolved_ids(ds_cfg)
            download_and_save(ds_cfg, resolved_ids)
        except Exception as e:
            print(f"  ERROR: {e}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
