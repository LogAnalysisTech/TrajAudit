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

import config
from utils import Failure, fold_traj, annotation_guidelines, extract_json

csv_lock = threading.Lock()

def analyze_agent_locate(failure:Failure, output_dir:str, runtime: int):
    FaultLocalization(failure, output_dir, runtime).run()

class StepOutput(BaseModel):
    query: list[dict] = [{}]
    thought: str = ""
    action: str = ""
    observation: str = ""
    done: bool = False
    tool_calls: list[str] = []

class FaultLocalization:
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
        while not step_output.done:
            step_output = self.step()
            self.save_trajectory()

    def step(self):
        step = StepOutput()
        step.query = deepcopy(self.history)
        thought, action = self.query(self.history)
        step.thought = thought
        step.action = action

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
            "content": f"Thought: {step.thought}\nAction: {step.action}",
        })

        self.history.append({
            "role": "user",
            "content": f"Observation: {step.observation}"
        })


    def handle_action(self, step: StepOutput):
        print("-----Handle Action-----")
        print(step.action)
        api_name = step.action.split(None, 1)[0]
        if api_name not in self.template_config["tools"]:
            step.observation = f"No tool called {api_name}"
        else:
            if api_name == "view_observation":
                index_id = int(step.action.split(None, 1)[1])
                if index_id < 0 or index_id >= len(self.folded_trajectory):
                    step.observation = f"api: {api_name}, attempt to visit step_id: {index_id}, but not exist in trajectory"
                else:
                    step.observation = json.loads(self.failure.original_traj[index_id])["observation"]
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
        instance_template = Template(self.template_config["agent"]["templates"]["instance_template"]).render(problem_statement=self.problem_statement,
                                                                                                             trajectories=self.folded_trajectory,
                                                                                                             test_code=self.failed_test_code,
                                                                                                             error_message=self.failed_test_error_description,
                                                                                                             annotation_guidelines=annotation_guidelines,
                                                                                                             apis=self.template_config["tools"],
                                                                                                             ground_truth=ground_truth,
                                                                                                             failure_insight=root_cause_prompt)
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
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": "respond",
                                "description": "Capture thoughts and actions in a structured way.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "thought": {
                                            "type": "string",
                                            "description": "your detailed reasoning process here"
                                        },
                                        "action": {
                                            "type": "string",
                                            "description": "api_call_here"
                                        }
                                    },
                                    "required": ["thought", "action"]
                                }
                            }
                        }
                    ],
                    tool_choice={"type": "function", "function": {"name": "respond"}}
                )

                message = answers.choices[0].message

                if message.tool_calls:
                    tool_call = message.tool_calls[0]
                    result = json.loads(tool_call.function.arguments)
                    thought = result["thought"]
                    action = result["action"]
                else:
                    raise ValueError(f"No tool call in response: {message.content}")

                self.token_consume += answers.usage.total_tokens
                print(f"thought: {thought}, action: {action}")
                return thought, action
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

        answers = self.client.chat.completions.create(
            model=config.MODEL,
            messages=messages,
            temperature=0,
            timeout=config.TIMEOUT,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "diagnose",
                        "description": "Identify the primary failure stage of the SE agent.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "stage": {
                                    "type": "string",
                                    "description": "Primary failure stage",
                                    "enum": [
                                        "Information Seeking",
                                        "Requirement Analysis",
                                        "Strategy Planning",
                                        "Code Implementation",
                                        "Validation & Feedback",
                                        "Context Management"
                                    ]
                                },
                                "reason": {
                                    "type": "string",
                                    "description": "Technical explanation of why this stage failed"
                                }
                            },
                            "required": ["stage", "reason"]
                        }
                    }
                }
            ],
            tool_choice={"type": "function", "function": {"name": "diagnose"}}
        )

        message = answers.choices[0].message

        if message.tool_calls:
            tool_call = message.tool_calls[0]
            result = json.loads(tool_call.function.arguments)
            self.stage_cause = result["reason"]
            self.stage = result["stage"]
            self.token_consume += answers.usage.total_tokens
            print(f"stage: {self.stage}, stage_cause: {self.stage_cause}")
