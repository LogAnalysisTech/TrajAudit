"""
Collect auto-code-rover reference trajectories for RootSE auto-code-rover instances
from s3://swe-bench-submissions (public, no-auth required).

Two runs collected:
  - acr-v2.0: verified/20241108_autocoderover-v2.0-claude-3-5-sonnet-20241022
  - acr-v2.1: verified/20250122_autocoderover-v2.1-claude-3-5-sonnet-20241022

Pass/fail determined from swe-bench/experiments results.json.

Step format matches RootSE auto-code-rover exactly:
  {index(str), thought(str), action(list), observation(str),
   response(str), execution_time(""), state(""), extra_info("")}

  action is a list of {func_name(str), arguments(dict), call_ok(bool)}

Usage:
  cd agent_system_log_normalization
  python baseline/FAMAS/collect_autocoderover_trajs.py
"""

import ast
import json
import os
import re
import subprocess
import tempfile

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INSTANCES = [
    "django__django-14999",
    "django__django-10914",
    "sympy__sympy-12419",
    "django__django-15252",
    "sympy__sympy-22714",
    "astropy__astropy-14182",
    "django__django-15695",
    "django__django-14787",
    "django__django-14534",
    "django__django-13551",
]

RUNS = {
    "acr-v2.0": "verified/20241108_autocoderover-v2.0-claude-3-5-sonnet-20241022",
    "acr-v2.1": "verified/20250122_autocoderover-v2.1-claude-3-5-sonnet-20241022",
}

S3_BUCKET = "swe-bench-submissions"
GH_REPO   = "swe-bench/experiments"
REF_BASE  = os.path.join(os.path.dirname(__file__),
                         "reference_trajs", "auto-code-rover", "swe-bench")

# Positional argument names per function (from v1 RootSE data)
FUNC_ARG_NAMES = {
    "search_class":            ["class_name"],
    "search_class_in_file":    ["class_name", "file_name"],
    "search_code":             ["code_str"],
    "search_code_in_file":     ["code_str", "file_name"],
    "search_method":           ["method_name"],
    "search_method_in_class":  ["method_name", "class_name"],
    "search_method_in_file":   ["method_name", "file_name"],
    "get_class_full_snippet":  ["class_name"],
    "get_code_around_line":    ["file_name", "line_number", "window_size"],
    "write_patch":             [],
}

# v2.1 uses "file_path" where RootSE uses "file_name"
V21_ARG_RENAME = {"file_path": "file_name"}

# User messages that are system prompts, not tool results (v2.0 only)
SYSTEM_PROMPT_PATTERNS = [
    "Let's analyze collected context",
    "Based on your analysis, answer below questions",
    "Write a patch for the issue",
    "You need to provide a patch",
    "Extracted a patch",
    "Ending the workflow",
]

# ---------------------------------------------------------------------------
# Pass/fail lookup from swe-bench/experiments
# ---------------------------------------------------------------------------

def get_resolved_set(run_path: str) -> set:
    """Fetch resolved instance IDs from swe-bench/experiments results.json."""
    import base64
    r = subprocess.run(
        ["gh", "api",
         f"repos/{GH_REPO}/contents/evaluation/{run_path}/results/results.json"],
        capture_output=True, text=True
    )
    content = json.loads(base64.b64decode(json.loads(r.stdout)["content"]).decode())
    return set(content.get("resolved", []))

# ---------------------------------------------------------------------------
# Common: call_ok refinement
# ---------------------------------------------------------------------------

def refine_call_ok(calls: list, observation: str) -> None:
    """Set call_ok=False for calls whose result shows no result."""
    no_result_patterns = [
        "No results found",
        "Could not find",
        "not found",
        "does not exist",
    ]
    for call in calls:
        fn = call["func_name"]
        pattern = re.compile(
            re.escape(f"Result of {fn}(") + r".*?\n(.*?)(?=Result of |\Z)",
            re.DOTALL | re.IGNORECASE
        )
        for m in pattern.finditer(observation):
            result_text = m.group(1)
            if any(p.lower() in result_text.lower() for p in no_result_patterns):
                call["call_ok"] = False
            break

# ---------------------------------------------------------------------------
# V2.0: python-block tool call extraction
# ---------------------------------------------------------------------------

def extract_tool_calls_v20(text: str) -> list:
    """Parse tool calls from ```python code blocks in assistant message (v2.0)."""
    calls = []
    blocks = re.findall(r"```python\s*(.*?)```", text, re.DOTALL)
    for block in blocks:
        for line in block.strip().split("\n"):
            line = line.strip()
            if not line or "(" not in line:
                continue
            try:
                tree = ast.parse(line, mode="eval")
            except SyntaxError:
                continue
            node = tree.body
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Name):
                continue
            func_name = func.id
            arg_names = FUNC_ARG_NAMES.get(func_name, [])
            arguments = {}
            for i, arg in enumerate(node.args):
                key = arg_names[i] if i < len(arg_names) else f"arg{i}"
                if isinstance(arg, ast.Constant):
                    arguments[key] = arg.value
                else:
                    try:
                        arguments[key] = ast.unparse(arg)
                    except Exception:
                        arguments[key] = str(arg)
            calls.append({
                "func_name": func_name,
                "arguments": arguments,
                "call_ok":   True,
            })
    return calls


def is_system_prompt(content: str) -> bool:
    return any(p in content for p in SYSTEM_PROMPT_PATTERNS)


def parse_conversation_v20(messages: list) -> list:
    """
    Convert v2.0 message list into raw steps [{response, action, observation}].
    Only assistant messages containing ```python blocks produce a step.
    """
    steps = []
    n = len(messages)
    i = 0
    while i < n:
        msg = messages[i]
        if msg.get("role") != "assistant":
            i += 1
            continue
        content = str(msg.get("content", ""))
        tool_calls = extract_tool_calls_v20(content)
        if not tool_calls:
            i += 1
            continue

        # Response text = everything before the first code block
        response = re.split(r"```python", content)[0].strip()

        # Observation = non-system user messages immediately following
        obs_parts = []
        j = i + 1
        while j < n and messages[j].get("role") == "user":
            uc = str(messages[j].get("content", ""))
            if not is_system_prompt(uc):
                obs_parts.append(uc)
            j += 1
        observation = "\n".join(obs_parts)

        refine_call_ok(tool_calls, observation)
        steps.append({
            "response":    response,
            "action":      tool_calls,
            "observation": observation,
        })
        i += 1

    return steps

# ---------------------------------------------------------------------------
# V2.1: native tool_calls format extraction
# ---------------------------------------------------------------------------

def parse_conversation_v21(messages: list) -> list:
    """
    Convert v2.1 message list (OpenAI tool_calls format) into raw steps.

    Pattern:
      [text assistant]?  → response text
      [tool_calls assistant]  → action (func_name from arguments["action"])
      [tool message]          → observation
    """
    steps = []
    n = len(messages)
    i = 0
    while i < n:
        msg = messages[i]
        tc_list = msg.get("tool_calls") or []
        if msg.get("role") != "assistant" or not tc_list:
            i += 1
            continue

        # Response = preceding text-only assistant message (if any)
        response = ""
        if (i > 0
                and messages[i - 1].get("role") == "assistant"
                and not (messages[i - 1].get("tool_calls") or [])):
            response = str(messages[i - 1].get("content", "") or "").strip()

        # Extract first tool call
        fn_data = tc_list[0].get("function", {})
        try:
            raw_args = json.loads(fn_data.get("arguments", "{}"))
        except Exception:
            raw_args = {}

        # Actual function name is in raw_args["action"]
        outer_name = fn_data.get("name", "unknown")
        action_val = raw_args.pop("action", outer_name)
        func_name  = action_val.replace(" ", "_")   # "list tree" → "list_tree"

        # Normalize v2.1 argument keys
        arguments = {V21_ARG_RENAME.get(k, k): v for k, v in raw_args.items()}

        # Observation = immediately following tool message
        observation = ""
        if i + 1 < n and messages[i + 1].get("role") == "tool":
            obs_raw = messages[i + 1].get("content", "")
            observation = obs_raw if isinstance(obs_raw, str) else str(obs_raw)
            i += 2
        else:
            i += 1

        call = {"func_name": func_name, "arguments": arguments, "call_ok": True}
        refine_call_ok([call], observation)

        steps.append({
            "response":    response,
            "action":      [call],
            "observation": observation,
        })

    return steps

# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_format(messages: list) -> str:
    """Return 'v21' if any message uses tool_calls; otherwise 'v20'."""
    for m in messages:
        if m.get("tool_calls"):
            return "v21"
    return "v20"

# ---------------------------------------------------------------------------
# Build final step list
# ---------------------------------------------------------------------------

def build_steps(cra_msgs: list, pa_msgs: list) -> list:
    """Combine context_retrieval and patching agents into final step list."""
    fmt = detect_format(cra_msgs)
    if fmt == "v21":
        steps = parse_conversation_v21(cra_msgs)
    else:
        steps = parse_conversation_v20(cra_msgs)

    # write_patch step: last text-only assistant message in patching_agent
    pa_text_assistants = [
        m for m in pa_msgs
        if m.get("role") == "assistant" and not (m.get("tool_calls") or [])
    ]
    if pa_text_assistants:
        patch_text = str(pa_text_assistants[-1].get("content", ""))
        steps.append({
            "response":    patch_text,
            "action":      [{"func_name": "write_patch", "arguments": {}, "call_ok": True}],
            "observation": "",
        })

    # Finalise into RootSE format (key order matches RootSE exactly)
    result = []
    for idx, s in enumerate(steps):
        result.append({
            "index":          str(idx),
            "thought":        "",
            "action":         s["action"],
            "observation":    s["observation"],
            "response":       s["response"],
            "execution_time": "",
            "state":          "",
            "extra_info":     "",
        })
    return result

# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def s3_download(s3_path: str, local_path: str) -> bool:
    r = subprocess.run(
        ["aws", "s3", "cp", s3_path, local_path, "--no-sign-request"],
        capture_output=True
    )
    return r.returncode == 0


def load_json_from_s3(s3_path: str, tmpdir: str, fname: str):
    local = os.path.join(tmpdir, fname)
    if s3_download(s3_path, local):
        return json.load(open(local, encoding="utf-8"))
    return None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Fetching pass/fail labels from swe-bench/experiments...\n")
    resolved = {}
    for label, run_path in RUNS.items():
        resolved[label] = get_resolved_set(run_path)
        print(f"  {label}: {sum(1 for i in INSTANCES if i in resolved[label])} PASS / "
              f"{sum(1 for i in INSTANCES if i not in resolved[label])} FAIL")
    print()

    total_saved = 0
    total_miss  = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for label, run_path in RUNS.items():
            print(f"=== {label} ({run_path}) ===")
            s3_base = f"s3://{S3_BUCKET}/{run_path}/trajs"

            for iid in INSTANCES:
                # Download context_retrieval_agent.json
                cra = load_json_from_s3(
                    f"{s3_base}/{iid}/attempt_0/context_retrieval_agent.json",
                    tmpdir, "cra.json"
                )
                # Download patching_agent.json
                pa = load_json_from_s3(
                    f"{s3_base}/{iid}/attempt_0/patching_agent.json",
                    tmpdir, "pa.json"
                )

                if cra is None or pa is None:
                    print(f"  [MISS] {iid}")
                    total_miss += 1
                    continue

                passed  = iid in resolved[label]
                steps   = build_steps(cra, pa)
                outcome = "success" if passed else "fail"

                dest_dir = os.path.join(REF_BASE, f"instance_{iid}", outcome)
                os.makedirs(dest_dir, exist_ok=True)
                dest_file = os.path.join(dest_dir, f"{label}.json")
                with open(dest_file, "w", encoding="utf-8") as f:
                    json.dump({"passed": passed, "steps": steps}, f,
                              ensure_ascii=False, indent=2)

                status = "PASS" if passed else "FAIL"
                print(f"  [{status}] {iid} → {outcome}/{label}.json ({len(steps)} steps)")
                total_saved += 1

            print()

    print(f"Done. Saved {total_saved}, missing {total_miss}.")


if __name__ == "__main__":
    main()
