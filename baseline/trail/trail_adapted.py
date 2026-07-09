"""
TRAIL adapted for RootSE benchmark.

Two-step pipeline:
  Step 1 — TRAIL original: feed the full trajectory to the LLM with TRAIL's error
            taxonomy and ask it to identify ALL error steps (step index as location).
  Step 2 — Annotation-guided selection: feed the identified errors + RootSE annotation
            guidelines to the LLM and ask it to pick the single earliest causal deviation.

Token consumption from BOTH calls is summed and reported.

Reference: Deshpande et al., "TRAIL: Trace Reasoning and Agentic Issue Localization",
           arXiv:2505.08638, PatronusAI 2025.
           https://github.com/patronus-ai/trail-benchmark (MIT License)
"""

import json
import os
import threading
import time

import pandas as pd
from json_repair import repair_json
from openai import OpenAI, APITimeoutError, RateLimitError

import config
from utils import Failure, annotation_guidelines, extract_json

csv_lock = threading.Lock()


def analyze_trail(failure: Failure, output_dir: str, runtime: int):
    TrailAdapted(failure, output_dir, runtime).run()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TRAIL_TAXONOMY = """
├── Reasoning Errors
│   ├── Hallucinations
│   │   ├── Language-only
│   │   └── Tool-related (fabricating tool outputs/capabilities)
│   ├── Information Processing
│   │   ├── Poor Information Retrieval (Tried to find information that was not relevant to the task)
│   │   └── Tool Output Misinterpretation (Made assumptions about the tool output or used the tool output in an incorrect context)
│   ├── Decision Making
│   │   ├── Incorrect Problem Identification (Misunderstood the overall task or the local task)
│   │   └── Tool Selection Errors (Used the wrong tool for the task)
│   └── Output Generation
│       ├── Formatting Errors (Errors with formatting and execution of code or structuring of output in a specific format)
│       └── Instruction Non-compliance (Failed to perform the task provided and instead did something else)
├── System Execution Errors
│   ├── Configuration
│   │   ├── Tool Definition Issues (The tool was not defined correctly by the user or contains some errors that make it inconsistent with its description)
│   │   └── Environment Setup Errors (includes permission problems and inability to access resources or API keys)
│   ├── API Issues
│   │   ├── Rate Limiting (Like 429)
│   │   ├── Authentication Errors (Like 401/403)
│   │   ├── Service Errors (Like 500)
│   │   └── Resource Not Found (Like 404)
│   └── Resource Management
│       ├── Resource Exhaustion (includes memory overflow)
│       └── Timeout Issues (The system took too long to respond)
└── Planning and Coordination Errors
     ├── Context Management
     │   ├── Context Handling Failures (includes window overflow and state tracking or forgetting important context)
     │   └── Resource Abuse (Called the tool excessively due to memory issues)
     └── Task Management
         ├── Goal Deviation (The system deviated from the task or the subtask)
         └── Task Orchestration (includes subtask coordination between agents and progress monitoring)
""".strip()


class TrailAdapted:
    def __init__(self, failure: Failure, output_dir: str, runtime: int):
        folder_path = os.path.join(
            output_dir, "TRAIL", config.MODEL,
            f"GROUND_TRUTH_{config.WITH_GROUND_TRUTH}",
            f"runtime_{runtime}"
        )
        self.filepath = os.path.join(folder_path, "result.csv")

        with csv_lock:
            os.makedirs(folder_path, exist_ok=True)

        self.failure = failure
        self.client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)
        self.token_consume = 0  # cumulative across both steps

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self):
        print(f"[TRAIL] {self.failure.instance_id}")

        # Step 1: TRAIL error identification
        errors = self._step1_identify_errors()

        if not errors:
            self._save_result("-1", "TRAIL Step 1 did not identify any errors")
            return

        # Step 2: Annotation-guided selection of earliest deviation
        failure_id, reason = self._step2_select_earliest(errors)
        self._save_result(failure_id, reason)

    # ------------------------------------------------------------------
    # Step 1 — TRAIL original: find ALL error steps
    # ------------------------------------------------------------------

    def _step1_identify_errors(self):
        trajectory_text = self._format_trajectory()

        ground_truth_section = ""
        if config.WITH_GROUND_TRUTH:
            ground_truth_section = (
                "\nThe following is a reference implementation. Note that multiple paths may "
                "achieve the task; differing from the gold patch does not necessarily mean "
                "the agent has deviated from a correct implementation.\n"
                f"<gold_patch>\n{self.failure.gold_patch}\n</gold_patch>\n"
            )

        prompt = f"""Follow the taxonomy below carefully and provide the output in the specified JSON format.

# Taxonomy
{TRAIL_TAXONOMY}

The agent system was given the following problem to solve:

<problem_description>
{self.failure.original_mission_prompt}
</problem_description>

The agent's execution trajectory (each step numbered from 0):

<trajectory>
{trajectory_text}
</trajectory>

The task failed to pass the predefined test:

<test_code>
{self.failure.failed_test_code}
</test_code>

<error_message>
{self.failure.failed_test_error_description}
</error_message>
{ground_truth_section}
Based on the taxonomy above, analyze the trajectory and identify ALL errors.

Rules:
- Use the integer step index (0, 1, 2, …) as the "step_index" field.
- Only include the final leaf subcategories from the taxonomy.
- For "Resource Abuse", mark the LAST instance. For all other errors, mark the FIRST instance.
- You must be exhaustive and find every error present.

Output strictly as raw JSON (no markdown, no extra text):
{{
    "errors": [
        {{
            "step_index": <integer>,
            "category": "<leaf taxonomy category>",
            "evidence": "<quoted text from the trajectory>",
            "description": "<detailed explanation of the error>",
            "impact": "<HIGH|MEDIUM|LOW>"
        }}
    ]
}}"""

        output = self._query([{"role": "user", "content": prompt}])
        if not output or "errors" not in output:
            return []
        errors = output["errors"]
        # Validate step indices
        n_steps = len(self.failure.original_traj)
        valid = []
        for e in errors:
            try:
                idx = int(e.get("step_index", -1))
                if 0 <= idx < n_steps:
                    e["step_index"] = idx
                    valid.append(e)
            except (ValueError, TypeError):
                continue
        print(f"[TRAIL] Step 1 found {len(valid)} errors")
        return valid

    # ------------------------------------------------------------------
    # Step 2 — annotation-guided selection of the earliest deviation
    # ------------------------------------------------------------------

    def _step2_select_earliest(self, errors):
        errors_text = json.dumps(errors, indent=2, ensure_ascii=False)

        prompt = f"""You are given a list of errors identified in a coding agent's trajectory, \
along with annotation guidelines for fault localization.

**Identified Errors from Step 1:**
{errors_text}

**Annotation Guidelines:**
{annotation_guidelines}

**Your Task:**
Based on the annotation guidelines above, select the ONE error that represents the \
**earliest deviation point** — the chronologically first step where the agent committed \
to a wrong path that is the direct ancestor of the final test failure.

Apply the guidelines strictly:
- Use the "But-For" test: if this step had been correct, would the failure still occur?
- Apply the Commitment Rule: distinguish cautious exploration from single-path commitment.
- Focus only on the causal chain leading to the test failure; ignore unrelated mistakes.

Output strictly as raw JSON (no markdown, no extra text):
{{
    "failure_id": <step_index as integer>,
    "reason": "<detailed technical explanation citing specific evidence from the trajectory>"
}}"""

        output = self._query([{"role": "user", "content": prompt}])
        if not output or "failure_id" not in output:
            return "-1", "TRAIL Step 2 failed to select earliest deviation"

        failure_id = str(output["failure_id"])
        reason = output.get("reason", "")
        print(f"[TRAIL] Step 2 selected failure_id={failure_id}")
        return failure_id, reason

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _format_trajectory(self):
        """Render trajectory as numbered steps. Truncate long observations."""
        lines = []
        for i, step_str in enumerate(self.failure.original_traj):
            try:
                step = json.loads(step_str)
            except json.JSONDecodeError:
                step = {"action": step_str, "observation": ""}

            thought = step.get("thought", "")
            action = step.get("action", "")
            observation = step.get("observation", "")

            # Serialize action compactly
            if isinstance(action, (dict, list)):
                action_str = json.dumps(action, ensure_ascii=False)
            else:
                action_str = str(action)

            obs_str = str(observation)

            lines.append(f"--- Step {i} ---")
            if thought:
                lines.append(f"Thought: {thought}")
            lines.append(f"Action: {action_str}")
            lines.append(f"Observation: {obs_str}")

        return "\n".join(lines)

    def _query(self, messages: list):
        """Query the LLM with retry logic; accumulate token usage."""
        try_time = 0
        while try_time < 5:
            try:
                response = self.client.chat.completions.create(
                    model=config.MODEL,
                    messages=messages,
                    temperature=config.TEMPERATURE,
                    timeout=config.TIMEOUT,
                    extra_body={"enable_thinking": False},
                )
                self.token_consume += response.usage.total_tokens
                content = response.choices[0].message.content
                print(f"[TRAIL] response: {content[:200]}...")
                return json.loads(repair_json(extract_json(content)))
            except (APITimeoutError, RateLimitError):
                print("[TRAIL] Rate limit / timeout, retrying...")
                try_time += 1
                time.sleep(60 * try_time)
            except Exception as e:
                err_str = str(e).lower()
                if "context" in err_str or "length" in err_str or "token" in err_str or "too long" in err_str:
                    print(f"[TRAIL] Context length exceeded, marking as failure: {e}")
                    return None
                print(f"[TRAIL] Exception: {e}")
                try_time += 1
        print("[TRAIL] All retries exhausted")
        return None

    def _save_result(self, failure_id: str, reason: str):
        new_data = pd.DataFrame([{
            "agent": self.failure.agent,
            "model": self.failure.model,
            "instance_id": f"{self.failure.agent}_{self.failure.model}_{self.failure.instance_id}",
            "failure_id": failure_id,
            "failure_reason": reason,
            "tokens": self.token_consume,   # total of Step 1 + Step 2
        }])
        with csv_lock:
            if os.path.exists(self.filepath):
                new_data.to_csv(self.filepath, mode="a", header=False, index=False)
            else:
                new_data.to_csv(self.filepath, index=False)
