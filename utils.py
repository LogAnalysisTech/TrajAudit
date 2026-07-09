import json
import re

from pydantic import BaseModel

import config

input_dir = './RootSE'
output_dir = './output'

annotation_guidelines = """
Objective: Identify the earliest step in the agent trajectory that deviates from the correct path, leading to the ultimate test failure.

Annotation Rules:

1. Dual-Layer Deviation Check (Thought & Action)
A deviation can occur in either the Mental Model or the Technical Execution. You must check them in this order:

Layer A: Intent/Planning (Thought): If the agent misinterprets the task, forms a flawed plan, or makes an incorrect inference in its "Thought" or "Response" process, this step is the earliest deviation point. Do not wait for a code change; the error has already started in the mind.

Layer B: Execution/Implementation (Action): If the agent’s "Thought" is correct (e.g., it identifies the right file and the right logic), but it fails to implement it correctly in the "Action" (e.g., writing buggy code, targeting the wrong line, or triggering irrecoverable environment damage), this step is the deviation point. 

* Summary: A step is a deviation if EITHER the reasoning is flawed OR the action fails to faithfully/correctly execute a sound reasoning.

2. The "Commitment" Rule (Distinguishing Exploration from Error)
Distinguish between "investigating possibilities" and "committing to a flawed direction":

Multi-Hypothesis (Exploration): If the agent proposes multiple possible causes or methods to investigate, it is exploring. Do not mark this as a deviation until the agent chooses and acts upon a flawed one.

Single-Path Commitment (Deviation): If the agent proposes only one specific method or plan and it is fundamentally flawed, this is a deviation, even if the agent uses tentative language (e.g., "may", "probably", "might", "I suspect"). The core criterion here is the "Alternative Test": if no other distinct path is proposed or pursued, the agent has effectively committed to that single direction. In this context, expressed uncertainty without the presence of alternative options still constitutes a single-path commitment.

Execution Omission: If the agent's "Thought" or "Response" states a need to gather information (e.g., "next I should check file X") but the "Action" skips this step, leading to a critical information gap, this step is a deviation.

3. Causal Root-Cause Only (Focus on the Lethal Error) Focus exclusively on the error that is the direct ancestor of the test failure.

Ignore Non-Fatal Mistakes: An agent may commit multiple mistakes throughout its trajectory, such as inefficient design, poor coding style, or "clumsy" incidental errors. You must disregard these if they do not directly contribute to the final test failure.

The "But-For" Test: Ask yourself: "If this specific step had been done correctly, would the final test failure still have occurred?" If the failure would have happened anyway due to a later, more fundamental error, then this step is not your deviation point.

Isolate the Causal Chain: Trace back strictly from the Test Error Description. Identify the chronologically first point that initiated the specific chain of events leading to that failure. Any error outside this causal chain is considered "noise" and should be ignored.

==============================================

Find the chronologically first step where the agent either:

Committed to a wrong mental model/plan.

Executed a wrong action.

Skipped a necessary information-gathering step.
"""

class Failure(BaseModel):
    instance_id: str
    agent: str
    model: str
    original_mission_prompt: str
    original_traj: list[str]
    gold_patch: str
    failed_test_code: str
    failed_test_error_description: str
    failed_test_expectation: str
    failure_id: str

def load_failure(absolute_path: str):
    with open(absolute_path, 'r') as f:
        data = json.load(f)
    failed_test_code = next(iter(data['failed_tests'].values()))["code"]
    failed_test_error_description = next(iter(data['failed_tests'].values()))["error_description"]
    failed_test_expectation = next(iter(data['failed_tests'].values()))["expected"]
    traj = []
    for content in data['original_traj']:
        traj.append(json.dumps(content, indent=2))
    failure = Failure(instance_id=data['instance_id'],agent=data['agent'], model=data['model'],original_mission_prompt=data['original_mission_prompt'],original_traj=traj,failure_id=data['failure_id'],failed_test_code=failed_test_code,failed_test_error_description=failed_test_error_description,failed_test_expectation=failed_test_expectation,gold_patch=data['gold_patch'])
    return failure

def fold_traj(traj: list[str], folded_item: list[str]):
    if not config.WITH_CONTEXT_AWARE_TRUNCATION:
        return traj

    patch_pattern = re.compile(r"--- [ab]/|\+\+\+ [ab]/|@@ -\d+,\d+ \+\d+,\d+ @@")
    error_keywords = ["exception", "fail", "error", "traceback", "invalid", "fatal"]

    folded_traj = []
    for content in traj:
        try:
            dict_content = json.loads(content)
        except json.JSONDecodeError:
            folded_traj.append(content)
            continue

        for item in folded_item:
            if item in dict_content and isinstance(dict_content[item], str):
                text = dict_content[item]
                length = len(text)
                text_lower = text.lower()

                if patch_pattern.search(text):
                    continue

                if length > 300:
                    dict_content[
                        item] = f"[Folded {length} characters. Please invoke API to see full context.]"
                else:
                    has_keyword = any(kw in text_lower for kw in error_keywords)

                    if has_keyword:
                        continue
                    else:
                        dict_content[item] = f"[Folded {length} chars (No errors/exceptions detected). Please invoke API to see full context.]"

        folded_traj.append(json.dumps(dict_content, indent=2, ensure_ascii=False))

    return folded_traj

def extract_json(text):
    """
    Extract the last complete, balanced JSON object from text.
    Uses a JSON-string-aware scanner so that braces inside string
    literals (e.g. '}}' in a thought field) are not mistaken for
    structural delimiters.

    Priority:
      1. Last balanced block that contains both "thought" and "action"
      2. Last balanced block (any)
      3. Legacy greedy-regex fallback
    """
    candidates = []
    depth = 0
    start = -1
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '"' and depth > 0:
            # Skip over a JSON string literal (handles \" escapes)
            i += 1
            while i < n:
                c = text[i]
                if c == '\\':
                    i += 2  # skip escaped character
                    continue
                if c == '"':
                    break   # end of string
                i += 1
        elif ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    candidates.append(text[start:i + 1])
        i += 1

    # Prefer the last candidate that contains both target keys
    for candidate in reversed(candidates):
        if '"thought"' in candidate and '"action"' in candidate:
            return candidate
    # Fall back to the last balanced block
    if candidates:
        return candidates[-1]
    # Last-resort: legacy greedy regex
    match = re.search(r'\{.*}', text, re.DOTALL)
    if match:
        return match.group()
    return "{" + text.strip().replace('\n', ', ').lower() + "}"