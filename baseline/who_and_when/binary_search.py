import os
import threading

import pandas as pd
from openai import OpenAI

import config
from utils import Failure, annotation_guidelines

csv_lock = threading.Lock()

class BinarySearchHelper:
    def __init__(self):
        super().__init__()
        self.total_tokens = 0

    def make_api_call(self, messages):
        client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)
        try:
            response = client.chat.completions.create(
                model=config.MODEL,
                messages=messages,
                temperature=config.TEMPERATURE,
                timeout=config.TIMEOUT
            )
            self.total_tokens += response.usage.total_tokens
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error during OpenAI API call: {e}")
            return None

    def format_list_for_llm(self, trajectory: list[str], start: int, end: int):
        trajectory = trajectory[start:end + 1]
        steps = "\n\n".join(trajectory)
        return f"<trajectory>\n{steps}\n</trajectory>"

    def construct_binary_search_prompt(self, failure, chat_segment_content, range_description, first_half_desc,
                                       second_half_desc):
        if config.WITH_GROUND_TRUTH:
            ground_truth = """

            - **Ground truth:**
                The following is a reference implementation. Note that multiple paths may achieve the task; differing from the gold patch does not necessarily mean the agent has deviated from a correct implementation.        
                <gold_patch>
                {gold_patch}
                </gold_patch>

            """.format(gold_patch=failure.gold_patch)
        else:
            ground_truth = ""

        prompt = """
            You are an expert Software Engineering Analyst. Your task is to pinpoint the **earliest deviation** in an agent's trajectory that eventually leads to a task failure.

            ### Context
            - **Problem Statement:** 
                {problem}
            - **Failed Test:** 
                <test_code>
                {test_code}
                </test_code>

                <error_message>
                {error_message}
                </error_message>
            {ground_truth}
            - **Annotation Guidelines:** 
                {annotation_guidelines}

            ### Trajectory Segment to Analyze ({range_description})
            {chat_segment_content}

            ### Task Instructions
            1. **Identify the Deviation:** Strictly follow the provided **Annotation Guidelines** to locate the very first step in this segment that qualifies as a deviation.
            2. **Binary Search Decision:** Determine which half of the current segment contains this first deviation step.
                - **First Half:** {first_half_desc}
                - **Second Half:** {second_half_desc}

            ### Constraints
            - Your output must contain **ONLY** the phrase "first half" or "second half".

            Result:
            """.format(problem=failure.original_mission_prompt, test_code=failure.failed_test_code,
                       error_message=failure.failed_test_error_description, annotation_guidelines=annotation_guidelines,
                       chat_segment_content=chat_segment_content, range_description=range_description,
                       first_half_desc=first_half_desc, second_half_desc=second_half_desc, ground_truth=ground_truth)

        return prompt

    def locate_failure_in_segment_recursive(self, failure: Failure, start: int, end: int):
        if start > end:
            return -1, f"unknown binary search error, start is greater than end, start:{start}, end:{end}"
        elif start == end:
            return start, ""

        mid = start + (end - start) // 2

        range_description = f"from index {start} to index {end}"
        first_half_desc = f"from index {start} to index {mid}"
        second_half_desc = f"from index {mid + 1} to index {end}"

        traj_segment = self.format_list_for_llm(failure.original_traj, start, end)

        prompt = self.construct_binary_search_prompt(failure, traj_segment, range_description, first_half_desc,
                                                second_half_desc)

        messages = [
            {"role": "system",
             "content": "You are an AI assistant specializing in localizing first deviation step in agent trajectory segments."},
            {"role": "user", "content": prompt}
        ]

        print(f"Analyzing step {start}-{end} for {failure.instance_id}...")

        result = self.make_api_call(messages)
        if not result:
            print(f"API call failed for segment {start}-{end}. Stopping binary search for {failure.instance_id}...")
            return -1, f"API call failed for segment {start}-{end}. Stopping binary search for {failure.instance_id}..."

        print(f"LLM Prediction for segment {start}-{end}: {result}")
        result_lower = result.lower()

        if "first half" in result_lower:
            return self.locate_failure_in_segment_recursive(failure, start, mid)
        elif "second half" in result_lower:
            new_start = min(mid + 1, end)
            return self.locate_failure_in_segment_recursive(failure, new_start, end)
        else:
            print(f"Warning: Ambiguous response '{result}' from LLM for segment {start}-{end}.")
            return -1, f"Warning: Ambiguous response '{result}' from LLM for segment {start}-{end}."

    def analyze_binary_search(self, failure: Failure, result_path: str, runtime: int):
        print(f"Evaluating {failure.instance_id} starts, runtime = {runtime} ...")

        folder_path = os.path.join(result_path, "binary_search", config.MODEL,
                                   f"GROUND_TRUTH_{config.WITH_GROUND_TRUTH}", f"runtime_{runtime}")
        filename = "result.csv"
        filepath = os.path.join(folder_path, filename)

        with csv_lock:
            os.makedirs(folder_path, exist_ok=True)

        print(f"========== Start Binary-Search Analysis on instance {failure.instance_id} ==========")
        failure_step, additional_info = self.locate_failure_in_segment_recursive(failure, 0, len(failure.original_traj) - 1)

        if failure_step == -1:
            print(f"========== End Binary-Search Analysis on instance {failure.instance_id} failed==========")
            print(additional_info)
        else:
            print(f"========== End Binary-Search Analysis on instance {failure.instance_id} succeeded==========")

        new_data = pd.DataFrame([{
            'agent': failure.agent,
            'model': failure.model,
            'instance_id': f"{failure.agent}_{failure.model}_{failure.instance_id}",
            'failure_id': str(failure_step),
            'failure_reason': additional_info,
            'tokens': self.total_tokens if failure_step != -1 else 0
        }])

        with csv_lock:
            if os.path.exists(filepath):
                new_data.to_csv(filepath, mode='a', header=False, index=False)
            else:
                new_data.to_csv(filepath, index=False)

        print(f"Evaluating {failure.instance_id} ends, runtime = {runtime} ...")

def analyze_binary_search(failure: Failure, result_path: str, runtime: int):
    BinarySearchHelper().analyze_binary_search(failure, result_path, runtime)