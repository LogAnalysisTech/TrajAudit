import json
import os
import threading
import time

import pandas as pd

from openai import OpenAI, APITimeoutError, RateLimitError

import config
from utils import Failure, annotation_guidelines, extract_json

csv_lock = threading.Lock()

def format_list_for_llm(trajectory: list[str], step_upper_limit: int):
    trajectory = trajectory[:step_upper_limit + 1]
    steps = "\n\n".join(trajectory)
    return f"<trajectory>\n{steps}\n</trajectory>"

def analyze_step_by_step(failure: Failure, result_path: str, runtime: int):
    print(f"Evaluating {failure.instance_id} starts, runtime = {runtime} ...")

    folder_path = os.path.join(result_path, "step_by_step", config.MODEL, f"GROUND_TRUTH_{config.WITH_GROUND_TRUTH}", f"runtime_{runtime}")
    filename = "result.csv"
    filepath = os.path.join(folder_path, filename)

    with csv_lock:
        os.makedirs(folder_path, exist_ok=True)

    # Resume support: skip if this instance was already completed
    instance_key = f"{failure.agent}_{failure.model}_{failure.instance_id}"
    if os.path.exists(filepath):
        existing = pd.read_csv(filepath)
        if instance_key in existing['instance_id'].values:
            print(f"Skipping {failure.instance_id} — already in results")
            return

    print(f"========== Start Step-by-Step Analysis on instance {failure.instance_id} ==========")

    system_prompt = "You are a helpful assistant in diagnosing and localizing failures in agent systems for software engineering."

    user_prompt = """
        The agent system was given the following problem to solve:

          <problem_description>
          {problem_statement}
          </problem_description>

          The system's execution trajectory up to the step {current_step} is shown below...

          <trajectories>
          {trajectories}
          </trajectories>

          The task failed to pass the predefined test.

          <test_code>
          {test_code}
          </test_code>

          <error_message>
          {error_message}
          </error_message>
          {ground_truth}
          Your task is to analyze execution trajectories and decide whether current step {current_step} is the earliest point where the agent's actions diverge from a successful implementation.

          **Annotation guidelines**:
          {annotation_guidelines}

          **Response format requirements**:
          You MUST respond in JSON format with exactly two keys:
          {{
            "current_step_failure": "output exactly one of: Yes / No / Passed. Yes = this step IS the first deviation. No = this step is not the deviation, keep looking. Passed = the failure step has already occurred earlier in the trajectory but you cannot pinpoint it; stop searching.",
            "failure_reason": "a detailed explanation of your decision"
          }}

          **When to output Passed**: If you have reviewed the trajectory up to the current step and are confident the deviation already occurred in a previous step but you cannot identify exactly which one, output Passed to avoid wasting further computation on later steps.

          **Critical**:
          - Your response must be valid JSON that can be parsed by json.loads()
          - Do NOT include markdown code blocks (no ```json ```)
          - Do NOT include any text before or after the JSON object
          - Invalid format will result in system error

        """

    client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)

    exception_record = ""

    api_error_signal = False

    total_token = 0

    for i in range(len(failure.original_traj)):
        print(f"Check instance {failure.instance_id}, Step {i}")
        if not config.WITH_GROUND_TRUTH:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.format(problem_statement=failure.original_mission_prompt,
                                                               trajectories=format_list_for_llm(failure.original_traj,
                                                                                                i),
                                                               test_code=failure.failed_test_code,
                                                               error_message=failure.failed_test_error_description,
                                                               current_step=i,
                                                               annotation_guidelines=annotation_guidelines,
                                                               ground_truth="\n")},
            ]
        else:
            gold_patch_message = """

                    The following is a reference implementation. Note that multiple paths may achieve the task; differing from the gold patch does not necessarily mean the agent has deviated from a correct implementation.        
                    <gold_patch>
                    {gold_patch}
                    </gold_patch>

                    """

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.format(problem_statement=failure.original_mission_prompt,
                                                               trajectories=format_list_for_llm(failure.original_traj,
                                                                                                i),
                                                               test_code=failure.failed_test_code,
                                                               error_message=failure.failed_test_error_description,
                                                               current_step=i,
                                                               annotation_guidelines=annotation_guidelines,
                                                               ground_truth=gold_patch_message.format(gold_patch=failure.gold_patch))},
            ]

        try_time = 0

        while try_time < 5:
            try:
                answers = client.chat.completions.create(
                    model=config.MODEL,
                    messages=messages,
                    temperature=config.TEMPERATURE,
                    timeout=config.TIMEOUT
                )

                response = [response.message.content for response in answers.choices][0]

                usage = answers.usage
                total_token += usage.total_tokens
                data = json.loads(extract_json(response))

                if 'Yes' in data['current_step_failure']:
                    new_data = pd.DataFrame([{
                        'agent': failure.agent,
                        'model': failure.model,
                        'instance_id': f"{failure.agent}_{failure.model}_{failure.instance_id}",
                        'failure_id': i,
                        'failure_reason': data['failure_reason'],
                        'tokens': total_token
                    }])
                    with csv_lock:
                        if os.path.exists(filepath):
                            new_data.to_csv(filepath, mode='a', header=False, index=False)
                        else:
                            new_data.to_csv(filepath, index=False)
                    print(f"========== End Step-by-Step Analysis on instance {failure.instance_id} Succeed ==========")
                    print(f"Evaluating {failure.instance_id} ends, runtime = {runtime} ...")
                    return
                elif 'Passed' in data['current_step_failure']:
                    # LLM believes the failure step has already been passed — stop early
                    new_data = pd.DataFrame([{
                        'agent': failure.agent,
                        'model': failure.model,
                        'instance_id': f"{failure.agent}_{failure.model}_{failure.instance_id}",
                        'failure_id': "-1",
                        'failure_reason': f"Early stop at step {i}: {data['failure_reason']}",
                        'tokens': total_token
                    }])
                    with csv_lock:
                        if os.path.exists(filepath):
                            new_data.to_csv(filepath, mode='a', header=False, index=False)
                        else:
                            new_data.to_csv(filepath, index=False)
                    print(f"========== Step-by-Step early stop at step {i} for {failure.instance_id} ==========")
                    print(f"Evaluating {failure.instance_id} ends, runtime = {runtime} ...")
                    return
                else:
                    break
            except (APITimeoutError, RateLimitError):
                exception_record += f"Try {try_time + 1} times by step {i}: API Timeout\n"
                print("API Timeout, retrying...")
                try_time += 1
                time.sleep(60 * try_time)
                if try_time >= 5:
                    api_error_signal = True
                continue
            except Exception as e:
                exception_record += f"Try {try_time + 1} times by step {i}: {e}\n"
                api_error_signal = True
                print("Exception :", e)
                break

    print(f"========== End Step-by-Step Analysis on instance {failure.instance_id} Fail ==========")

    if api_error_signal:
        failure_reason = f"no result due to API error, exception record: \n{exception_record}"
    else:
        failure_reason = f"step-by-step fails to locate a failure step after checking the whole traj"

    new_data = pd.DataFrame([{
        'agent': failure.agent,
        'model': failure.model,
        'instance_id': f"{failure.agent}_{failure.model}_{failure.instance_id}",
        'failure_id': "-1",
        'failure_reason': failure_reason,
        'tokens': 0
    }])

    with csv_lock:
        if os.path.exists(filepath):
            new_data.to_csv(filepath, mode='a', header=False, index=False)
        else:
            new_data.to_csv(filepath, index=False)

    print(f"Evaluating {failure.instance_id} ends, runtime = {runtime} ...")