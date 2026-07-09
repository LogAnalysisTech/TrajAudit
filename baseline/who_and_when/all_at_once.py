import json
import os
import threading
import time
import pandas as pd

from openai import OpenAI, APITimeoutError, RateLimitError

import config
from utils import Failure, annotation_guidelines, extract_json

csv_lock = threading.Lock()

def format_list_for_llm(trajectory: list[str]):
    steps = "\n\n".join(trajectory)
    return f"<trajectory>\n{steps}\n</trajectory>"

def analyze_all_at_once(failure: Failure, result_path: str, runtime: int):
    print(f"Evaluating {failure.instance_id} starts, runtime = {runtime} ...")
    folder_path = os.path.join(result_path, "all_at_once", config.MODEL, f"GROUND_TRUTH_{config.WITH_GROUND_TRUTH}", f"runtime_{runtime}")
    filename = f"result.csv"
    filepath = os.path.join(folder_path, filename)

    with csv_lock:
        os.makedirs(folder_path, exist_ok=True)

    print(f"========== Start All-at-Once Analysis on instance {failure.instance_id} ==========")

    system_prompt = "You are a helpful assistant in diagnosing and localizing failures in agent systems for software engineering."

    user_prompt = """
    The agent system was given the following problem to solve:

      <problem_description>
      {problem_statement}
      </problem_description>

      The system's execution trajectory is shown below...

      <trajectories>
      {trajectories}
      </trajectories>

      However, the task failed to pass the predefined test.

      <test_code>
      {test_code}
      </test_code>
      
      <error_message>
      {error_message}
      </error_message>
      {ground_truth}
      **Annotation guidelines:**:
      {annotation_guidelines}

      **Response format requirements**:
      You MUST respond in JSON format with exactly two keys:
      {{
        "failure_id": "the index of the first deviation step from a successful implementation",
        "failure_reason": "a detailed explanation of why this specific step caused the task to fail, including what went wrong and how it led to the failure"
      }}

      **Critical**:
      - Your response must be valid JSON that can be parsed by json.loads()
      - Do NOT include markdown code blocks (no ```json ```)
      - Do NOT include any text before or after the JSON object
      - Invalid format will result in system error
    
    """

    if not config.WITH_GROUND_TRUTH:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(problem_statement=failure.original_mission_prompt,
                                                           trajectories=format_list_for_llm(failure.original_traj),
                                                           test_code=failure.failed_test_code,
                                                           error_message=failure.failed_test_error_description,
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
                                                           trajectories=format_list_for_llm(failure.original_traj),
                                                           test_code=failure.failed_test_code,
                                                           error_message=failure.failed_test_error_description,
                                                           annotation_guidelines=annotation_guidelines,
                                                           ground_truth=gold_patch_message.format(gold_patch=failure.gold_patch))},
        ]

    client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)

    try_time = 0

    exception_record = ""

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
            data = json.loads(extract_json(response))
            new_data = pd.DataFrame([{
                'agent': failure.agent,
                'model': failure.model,
                'instance_id': f"{failure.agent}_{failure.model}_{failure.instance_id}",
                'failure_id': data['failure_id'],
                'failure_reason': data['failure_reason'],
                'tokens': usage.total_tokens
            }])

            with csv_lock:
                if os.path.exists(filepath):
                    new_data.to_csv(filepath, mode='a', header=False, index=False)
                else:
                    new_data.to_csv(filepath, index=False)
            print(f"========== End All-at-Once Analysis on instance {failure.instance_id} Succeed ==========")
            return

        except (APITimeoutError, RateLimitError):
            exception_record += f"Try {try_time + 1} times: API Timeout\n"
            print("API Timeout, retrying...")
            try_time += 1
            time.sleep(60 * try_time)
            continue
        except Exception as e:
            exception_record += f"Try {try_time + 1} times: {e}\n"
            print("Exception :", e)
            break

    print(f"========== End All-at-Once Analysis on instance {failure.instance_id} Fail ==========")

    new_data = pd.DataFrame([{
        'agent': failure.agent,
        'model': failure.model,
        'instance_id': f"{failure.agent}_{failure.model}_{failure.instance_id}",
        'failure_id': "-1",
        'failure_reason': f"no result due to API error, exception record: \n{exception_record}",
        'tokens': 0
    }])

    with csv_lock:
        if os.path.exists(filepath):
            new_data.to_csv(filepath, mode='a', header=False, index=False)
        else:
            new_data.to_csv(filepath, index=False)

    print(f"Evaluating {failure.instance_id} ends, runtime = {runtime} ...")