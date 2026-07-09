"""
Collect OpenHands reference trajectories for RootSE openhands instances
from nebius/SWE-rebench-openhands-trajectories (HuggingFace).

For each of the 30 RootSE openhands instances, fetches ALL available
trajectories from the dataset and saves them under:

  reference_trajs/openhands/swe-rebench/{instance_id}/success/{trajectory_id}.json
  reference_trajs/openhands/swe-rebench/{instance_id}/fail/{trajectory_id}.json

Step format matches RootSE openhands exactly:
  {index (int), thought (str), action (dict), observation (str), response (str)}

Usage:
  cd agent_system_log_normalization
  python baseline/FAMAS/collect_openhands_trajs.py
"""

import json
import os

from datasets import load_dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUR_INSTANCES = [
    "12rambau__sepal_ui-646",
    "15five__scim2-filter-parser-13",
    "ARMmbed__mbed-tools-138",
    "ARMmbed__mbed-tools-91",
    "AzureAD__microsoft-authentication-library-for-python-530",
    "Azure__azure-cli-1876",
    "BAMWelDX__weldx-384",
    "Bachmann1234__diff_cover-413",
    "BoboTiG__ebook-reader-dict-2101",
    "BoboTiG__ebook-reader-dict-2133",
    "BoboTiG__ebook-reader-dict-2145",
    "BoboTiG__ebook-reader-dict-2149",
    "CODAIT__exchange-metadata-converter-25",
    "CODAIT__exchange-metadata-converter-38",
    "CORE-GATECH-GROUP__serpent-tools-377",
    "CS-SI__eodag-634",
    "David-Wobrock__sqlvalidator-43",
    "Delgan__loguru-1236",
    "Delgan__loguru-1299",
    "Dorthu__openapi3-57",
    "Duke-GCB__lando-169",
    "Duke-GCB__lando-173",
    "ESSS__conda-devenv-51",
    "ESSS__oop-ext-21",
    "Erotemic__ubelt-64",
    "Fatal1ty__mashumaro-116",
    "Fatal1ty__mashumaro-130",
    "FirefighterBlu3__python-pam-32",
    "GjjvdBurg__paper2remarkable-43",
    "ansible__ansible-builder-336",
]

REF_BASE = os.path.join(
    os.path.dirname(__file__),
    "reference_trajs", "openhands", "swe-rebench"
)

# ---------------------------------------------------------------------------
# Conversion: nebius trajectory messages → RootSE openhands step format
# ---------------------------------------------------------------------------

def convert_messages_to_steps(messages: list) -> list:
    """
    Convert nebius OpenHands message list to RootSE openhands step format.

    Message pattern (nebius):
      [0] system
      [1] user  — task description
      [2] assistant — response text + tool_calls[{function:{name,arguments}}]
      [3] tool  — tool output (observation)
      [4] assistant — ...
      ...

    RootSE openhands step format:
      {
        "index":       0,              # int
        "thought":     "",             # always empty in openhands RootSE
        "action":      {"tool": "str_replace_editor", "input": {...}},
        "observation": "...",
        "response":    "..."           # assistant text before the tool call
      }
    """
    steps = []
    step_index = 0
    i = 2  # skip system + initial user message

    while i < len(messages):
        msg = messages[i]

        if msg.get("role") == "assistant":
            response_text = msg.get("content", "") or ""
            tool_calls = msg.get("tool_calls", []) or []

            # Build action dict from first tool call
            action = {}
            if tool_calls:
                func = tool_calls[0].get("function", {})
                tool_name = func.get("name", "unknown")
                raw_args = func.get("arguments", "{}")
                try:
                    tool_input = json.loads(raw_args)
                except Exception:
                    tool_input = {"raw": raw_args}
                action = {"tool": tool_name, "input": tool_input}

            # Next tool message = observation
            observation = ""
            if i + 1 < len(messages) and messages[i + 1].get("role") == "tool":
                obs_raw = messages[i + 1].get("content", "")
                observation = obs_raw if isinstance(obs_raw, str) else str(obs_raw)
                i += 2
            else:
                i += 1

            # Only record steps that have an action (skip pure-text assistant turns)
            if action:
                steps.append({
                    "index":       step_index,
                    "thought":     "",
                    "action":      action,
                    "observation": observation,
                    "response":    response_text,
                })
                step_index += 1
        else:
            i += 1

    return steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading nebius/SWE-rebench-openhands-trajectories (streaming)...")
    ds = load_dataset(
        "nebius/SWE-rebench-openhands-trajectories",
        split="train",
        streaming=True,
    )

    target_set = set(OUR_INSTANCES)
    collected = {inst: [] for inst in OUR_INSTANCES}

    print(f"Scanning dataset for {len(OUR_INSTANCES)} instances...\n")

    for row in ds:
        iid = row["instance_id"]
        if iid not in target_set:
            continue
        collected[iid].append(row)
        # Print progress
        total_found = sum(len(v) for v in collected.values())
        remaining = sum(1 for v in collected.values() if len(v) == 0)
        if total_found % 10 == 0 or remaining == 0:
            print(f"  Found so far: {total_found} trajectories, "
                  f"{remaining} instances still not seen")
        if remaining == 0:
            # Could stop early, but keep scanning for more trajectories
            pass

    print("\nScanning complete. Saving files...\n")

    total_saved = 0
    for inst in OUR_INSTANCES:
        rows = collected[inst]
        if not rows:
            print(f"  [MISSING] {inst}: no trajectories found in dataset")
            continue

        for row in rows:
            passed = bool(row["resolved"])
            traj_id = row.get("trajectory_id", f"traj_{rows.index(row)}")
            steps = convert_messages_to_steps(row["trajectory"])

            outcome_dir = "success" if passed else "fail"
            dest_dir = os.path.join(REF_BASE, inst, outcome_dir)
            os.makedirs(dest_dir, exist_ok=True)

            # Use trajectory_id as filename (sanitise slashes)
            safe_name = traj_id.replace("/", "_").replace("\\", "_")
            dest_file = os.path.join(dest_dir, f"{safe_name}.json")
            with open(dest_file, "w", encoding="utf-8") as f:
                json.dump({"passed": passed, "steps": steps}, f,
                          ensure_ascii=False, indent=2)

            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {inst} → {outcome_dir}/{safe_name}.json  "
                  f"({len(steps)} steps)")
            total_saved += 1

    print(f"\nDone. Saved {total_saved} reference trajectories.")


if __name__ == "__main__":
    main()
