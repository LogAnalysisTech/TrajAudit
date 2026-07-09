import json
import os
import threading
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml
from jinja2 import Template
from openai import OpenAI, APITimeoutError, RateLimitError
from pydantic import BaseModel
from json_repair import repair_json

import config
from utils import Failure, fold_traj, annotation_guidelines, extract_json

csv_lock = threading.Lock()

def analyze_agent_locate_claude(failure:Failure, output_dir:str, runtime: int):
    FaultLocalizationClaude(failure, output_dir, runtime).run()

class StepOutput(BaseModel):
    query: list[dict] = [{}]
    thought: str = ""
    action: str = ""
    observation: str = ""
    done: bool = False
    tool_calls: list[str] = []

class FaultLocalizationClaude:
    def __init__(self, failure: Failure, result_path: str, runtime: int):
        super().__init__()

        folder_path = os.path.join(result_path, "agent_locate", config.MODEL, f"GROUND_TRUTH_{config.WITH_GROUND_TRUTH}_DIAGNOSIS_{config.WITH_DIAGNOSIS}_CONTEXT_TRUNCATION_{config.WITH_CONTEXT_AWARE_TRUNCATION}", f"runtime_{runtime}")
        filename = "result.csv"
        self.filepath = os.path.join(folder_path, filename)

        with csv_lock:
            os.makedirs(folder_path, exist_ok=True)

        self.failure = failure
        self.client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)
        self.problem_statement = failure.original_mission_prompt
        self.folded_trajectory = fold_traj(failure.original_traj, ["observation"])
        self.failed_test_code = failure.failed_test_code
        self.failed_test_error_description = failure.failed_test_error_description
        self.gold_patch = failure.gold_patch
        self.annotation_guidelines = annotation_guidelines
        self.history = []
        self.trajectory = []
        self.traj_path = Path(folder_path) / f"{self.failure.agent}_{self.failure.model}_{self.failure.instance_id}.traj"
        self.token_consume = 0
        self.stage = ""
        self.stage_cause = ""

        with open('./TrajAudit/template.yaml', 'r') as f:
            self.template_config = yaml.safe_load(f)

        if config.WITH_DIAGNOSIS:
            self.check_root_cause()

        print(f"Setting up agent for problem {failure.instance_id}")

        self.add_system_message_to_history()
        self.add_user_message_to_history()

    def run(self):
        step_output = StepOutput()
        step_count = 0
        # When full trajectory is visible (no folding), no interaction is needed:
        # the agent should submit directly in one step, mirroring all_at_once + diagnosis.
        max_steps = 1 if not config.WITH_CONTEXT_AWARE_TRUNCATION else 5
        while not step_output.done:
            if step_count >= max_steps:
                break
            if step_count == max_steps - 1:
                # Final step: force the model to submit
                self.history.append({
                    "role": "user",
                    "content": f"[SYSTEM CONSTRAINT] This is your FINAL step ({max_steps}/{max_steps}). You MUST call submit_result now with your best conclusion based on what you have seen so far. Do NOT call view_observation."
                })
            step_output = self.step()
            self.save_trajectory()
            step_count += 1

    def step(self):
        step = StepOutput()
        step.query = deepcopy(self.history)
        output = self.query(self.history)
        if not output or ("thought" not in output) or ("action" not in output):
            step.done = True
            new_data = pd.DataFrame([{
                'agent': self.failure.agent,
                'model': self.failure.model,
                'instance_id': f"{self.failure.agent}_{self.failure.model}_{self.failure.instance_id}",
                'failure_id': "-1",
                'failure_reason': "llm output is not a valid JSON",
                'tokens': 0
            }])
            with csv_lock:
                if os.path.exists(self.filepath):
                    new_data.to_csv(self.filepath, mode='a', header=False, index=False)
                else:
                    new_data.to_csv(self.filepath, index=False)
            step.thought = ""
            step.action = ""
            return step
        step.thought = output["thought"]
        step.action = output["action"]

        step = self.handle_action(step)
        self.add_step_to_history(step)
        self.add_step_to_trajectory(step)

        return step

    def add_step_to_trajectory(self, step: StepOutput):
        self.trajectory.append({
            "action": step.action,
            "thought": step.thought,
            "observation": step.observation,
            "query": step.query
        })


    def add_step_to_history(self, step: StepOutput):
        self.history.append({
            "role": "assistant",
            "content": json.dumps({"thought": step.thought, "action": step.action}),
        })

        self.history.append({
            "role": "user",
            "content": f"Observation: {step.observation}"
        })


    @staticmethod
    def _summarize_action(action, head=40, tail=20) -> str:
        """Return a compact one-line summary of an action, truncating if needed."""
        if isinstance(action, dict):
            tool = action.get("tool", action.get("func_name", ""))
            inp  = action.get("input", action.get("arguments", {}))
            # Pull a representative path/name from the input dict
            hint = (inp.get("path") or inp.get("class_name") or
                    inp.get("method_name") or inp.get("file_name") or
                    next(iter(inp.values()), "") if inp else "")
            raw = f"{tool}: {hint}" if hint else tool
        elif isinstance(action, list):
            # auto-code-rover style: list of tool calls
            names = [a.get("func_name", "") for a in action if isinstance(a, dict)]
            raw = ", ".join(names[:3]) + ("…" if len(names) > 3 else "")
        else:
            raw = str(action).replace("\n", " ")

        raw = raw.strip()
        if len(raw) <= head + tail + 3:
            return raw
        return f"{raw[:head]}...{raw[-tail:]}"

    def handle_action(self, step: StepOutput):
        print("-----Handle Action-----")
        print(step.action)
        api_name = step.action.split(None, 1)[0]
        if api_name not in self.template_config["tools"]:
            step.observation = f"No tool called {api_name}"
        else:
            if api_name == "view_observation":
                # Support batch lookup: view_observation 1 2 3
                raw_indices = step.action.split(None, 1)[1].split()
                parts = []
                for idx_str in raw_indices:
                    try:
                        index_id = int(idx_str)
                    except ValueError:
                        parts.append(f"[Step {idx_str}: invalid index]")
                        continue
                    if index_id < 0 or index_id >= len(self.folded_trajectory):
                        parts.append(f"[Step {index_id}: not found in trajectory]")
                    else:
                        traj_step = json.loads(self.failure.original_traj[index_id])
                        obs = traj_step["observation"]
                        if config.WITH_CONTEXT_AWARE_TRUNCATION and len(obs) > 2000:
                            obs = obs[:2000] + f"\n...[truncated, {len(obs)} chars total]"
                        action_summary = self._summarize_action(traj_step.get("action", ""))
                        parts.append(f"=== Step {index_id} ({action_summary}) ===\n{obs}")
                step.observation = "\n\n".join(parts)
            elif api_name == "submit_result":
                params = step.action.split(None, 1)[1]
                failure_index_in_trajectory = params.split(None, 1)[0]
                explanation = params.split(None, 1)[1]
                step.observation = f"""Final Conclusion:
                1. failed step: {failure_index_in_trajectory}
                2. Explanation: {explanation}"""
                step.done = True
                new_data = pd.DataFrame([{
                    'agent': self.failure.agent,
                    'model': self.failure.model,
                    'instance_id': f"{self.failure.agent}_{self.failure.model}_{self.failure.instance_id}",
                    'failure_id': str(failure_index_in_trajectory),
                    'failure_reason': explanation,
                    'tokens': self.token_consume
                }])

                with csv_lock:
                    if os.path.exists(self.filepath):
                        new_data.to_csv(self.filepath, mode='a', header=False, index=False)
                    else:
                        new_data.to_csv(self.filepath, index=False)
        return step

    def add_system_message_to_history(self):
        self.history.append({"role": "system", "content": self.template_config["agent"]["templates"]["system_template"]})

    def add_user_message_to_history(self):
        if config.WITH_GROUND_TRUTH:
            ground_truth = """
The following is a reference implementation. Note that multiple paths may achieve the task; differing from the gold patch does not necessarily mean the agent has deviated from a correct implementation.        
<gold_patch>
{gold_patch}
</gold_patch>
            """.format(gold_patch=self.gold_patch)
        else:
            ground_truth = ""
        if self.stage and self.stage_cause:
            root_cause_prompt = """
**Diagnostic Prior (Heuristic Guidance)**:
Preliminary analysis of the test code and failure symptoms indicates that the root cause most likely originated during the **{stage}** stage.

**Reasoning Insight**:
{failure_insight}

**Localization Strategy**:
Prioritize your audit on this specific phase. You must backtrack through the reasoning trajectory to identify the **Earliest Deviation Point** within the {stage} context. Look for where the agent's intent first misaligned with the technical requirements or the repository state.
""".format(stage=self.stage, failure_insight=self.stage_cause)
        else:
            root_cause_prompt = ""
        if config.WITH_CONTEXT_AWARE_TRUNCATION:
            step_budget_hint = "- Step budget: You have a maximum of 5 steps total. Plan your investigation efficiently. Prioritize batch view_observation calls (e.g., view_observation 3 7 12) to inspect multiple steps at once. Reserve your final step for submit_result."
        else:
            step_budget_hint = "- All trajectory observations are fully visible above. You must call submit_result directly in this single step."

        instance_template = Template(self.template_config["agent"]["templates"]["instance_template"]).render(problem_statement=self.problem_statement,
                                                                                                             trajectories=self.folded_trajectory,
                                                                                                             test_code=self.failed_test_code,
                                                                                                             error_message=self.failed_test_error_description,
                                                                                                             annotation_guidelines=annotation_guidelines,
                                                                                                             apis=self.template_config["tools"],
                                                                                                             ground_truth=ground_truth,
                                                                                                             failure_insight=root_cause_prompt,
                                                                                                             step_budget_hint=step_budget_hint)
        self.history.append({"role": "user", "content": instance_template})

    def save_trajectory(self):
        self.traj_path.write_text(json.dumps(self.trajectory, indent=2))

    def query(self, history: list[dict[str, str]]):

        try_time = 0

        while try_time < 5:
            try:
                if try_time == 0:
                    curr_temperature = config.TEMPERATURE
                else:
                    curr_temperature = 0.1
                answers = self.client.chat.completions.create(
                    model=config.MODEL,
                    messages=history,
                    temperature=curr_temperature,
                    timeout=config.TIMEOUT,
                    extra_body={
                        "enable_thinking": False
                    }
                )
                self.token_consume += answers.usage.total_tokens
                response = [response.message.content for response in answers.choices][0]
                print(f"response: {response}")
                output = json.loads(repair_json(extract_json(response)))
                return output
            except (APITimeoutError, RateLimitError):
                print("API Timeout, retrying...")
                try_time += 1
                time.sleep(60 * try_time)
                continue
            except Exception as e:
                print("Exception :", e)
                try_time += 1
                continue
        print("API retry time reach limits, return null")
        return ""

    def check_root_cause(self):

        instance_template = Template(self.template_config["agent"]["templates"]["diagnosis_template"]).render(
            problem_statement=self.problem_statement,
            test_code=self.failed_test_code,
            error_message=self.failed_test_error_description)

        system_prompt = """
Role: SE Failure Diagnostic Engine.
Task: Identify the primary failure stage from: 
[Information Seeking, Requirement Analysis, Strategy Planning, Code Implementation, Validation & Feedback, Context Management].

Output: Strict JSON {"stage": "...", "reason": "..."}. 
Requirements: Technical precision, no preamble, raw JSON only.
"""

        messages = [
            {"role": "system",
             "content": system_prompt},
            {"role": "user", "content": instance_template}
        ]

        output = self.query(messages)

        self.stage_cause = output["reason"]
        self.stage = output["stage"]
