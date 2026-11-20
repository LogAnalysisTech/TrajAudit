import argparse
import json
import os.path
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
import pandas as pd

import config
from baseline.who_and_when.all_at_once import analyze_all_at_once
from baseline.who_and_when.binary_search import analyze_binary_search
from baseline.who_and_when.step_by_step import analyze_step_by_step
from baseline.trail.trail_adapted import analyze_trail
from baseline.FAMAS.famas import analyze_famas
from TrajAudit.fault_localization import analyze_agent_locate
from TrajAudit.fault_localization_claude import analyze_agent_locate_claude
from TrajAudit.fault_localization_generic import analyze_agent_locate_generic
from utils import input_dir, load_failure, output_dir

def count_json_files(path):
    count = 0
    for root, dirs, files in os.walk(path):
        count += sum(1 for f in files if f.endswith('.json'))
    return count

def run_all_cases(baseline_type: str):
    # binary_search makes ~6-7 recursive API calls per case → 1 worker to avoid rate limit.
    # step_by_step makes N calls per case (one per step) → 5 workers as a safe middle ground.
    if baseline_type == "binary_search":
        workers = 1
    elif baseline_type == "step_by_step":
        workers = 5
    else:
        workers = 32
    with ThreadPoolExecutor(max_workers=workers) as step_executor:
        for runtime in range(1, config.RUN_TIME + 1):
            try:
                p = Path(input_dir)
                for file in p.rglob("*.json"):

                    failure = load_failure(str(file.resolve()))

                    if baseline_type == "all_at_once":
                        step_executor.submit(analyze_all_at_once, failure, output_dir, runtime)
                    elif baseline_type == "step_by_step":
                        step_executor.submit(analyze_step_by_step, failure, output_dir, runtime)
                    elif baseline_type == "binary_search":
                        step_executor.submit(analyze_binary_search, failure, output_dir, runtime)
                    elif baseline_type == "trail":
                        step_executor.submit(analyze_trail, failure, output_dir, runtime)
                    elif baseline_type == "FAMAS":
                        step_executor.submit(analyze_famas, failure, output_dir, runtime)
                    elif baseline_type == "agent_locate":
                        if "claude" in config.MODEL.lower():
                            step_executor.submit(analyze_agent_locate_claude, failure, output_dir, runtime)
                        else:
                            step_executor.submit(analyze_agent_locate_generic, failure, output_dir, runtime)

            except Exception as e:
                print(e)

def get_result_dir(baseline_type: str, runtime: int) -> str:
    # TRAIL writes to output/TRAIL (see baseline/trail/trail_adapted.py); keep the
    # directory name case-exact so it also works on case-sensitive filesystems.
    dir_name = "TRAIL" if baseline_type == "trail" else baseline_type
    if baseline_type == "agent_locate":
        setting = f"GROUND_TRUTH_{config.WITH_GROUND_TRUTH}_DIAGNOSIS_{config.WITH_DIAGNOSIS}_CONTEXT_TRUNCATION_{config.WITH_CONTEXT_AWARE_TRUNCATION}"
    else:
        setting = f"GROUND_TRUTH_{config.WITH_GROUND_TRUTH}"
    return os.path.join(output_dir, dir_name, config.MODEL, setting, f"runtime_{runtime}")

def evaluate_all_cases(baseline_type: str, eval_only: bool = False):

    if not eval_only:
        run_all_cases(baseline_type)

    # Avg_Tokens: mean over every row that has a result, counted once per setting
    # (independent of tolerance). Denominator is the number of rows with results,
    # which can be smaller than Total when some cases are missing.
    dfs = {runtime: pd.read_csv(os.path.join(get_result_dir(baseline_type, runtime), "result.csv"))
           for runtime in range(1, config.RUN_TIME + 1)}
    token_series = pd.concat([df['tokens'] for df in dfs.values()]).dropna()
    avg_tokens = token_series.sum() / len(token_series) if len(token_series) else 0

    for tolerance in config.TOLERANCE:
        all_success = 0
        all_total = 0

        for runtime in range(1, config.RUN_TIME + 1):
            df = dfs[runtime]
            result_dir = get_result_dir(baseline_type, runtime)
            total, success = count_json_files("./RootSE/"), 0
            all_total += total
            failure_cases = []

            for index, row in df.iterrows():
                agent = row['agent']
                model = row['model']
                instance_id = row["instance_id"].split('_', 2)[-1]
                failure_id = row["failure_id"]
                failure = load_failure(os.path.abspath(f"./RootSE/{agent}/{model}/{instance_id}.json"))
                if failure and int(failure_id) != -1 and (abs(int(failure.failure_id) - int(failure_id)) <= tolerance):
                    success += 1
                    all_success += 1
                else:
                    curr_case = {"instance_id": instance_id, "ground_truth_failure_id": failure.failure_id,
                             "forecast_failure_id": failure_id, "forecast_reason": row["failure_reason"]}
                    failure_cases.append(curr_case)

            summary = {"success_rate": str(success/total), "total_cases": str(total)}
            failure_cases.append(summary)

            with open(os.path.join(result_dir, f"eval_tolerance_{tolerance}.json"), "w") as f:
                json.dump(failure_cases, f, indent=4)

        if baseline_type == "agent_locate":
            log_line = f"Baseline: {baseline_type}, Model: {config.MODEL}, WithGroundTruth: {config.WITH_GROUND_TRUTH}, WithDiagnosisL {config.WITH_DIAGNOSIS}, WithContextAwareTruncation: {config.WITH_CONTEXT_AWARE_TRUNCATION}, Tolerance: {tolerance}, Total: {all_total}, Success: {all_success}, Avg_Accuracy: {all_success/all_total:.4f}, Avg_Tokens: {avg_tokens}\n"
        else:
            log_line = f"Baseline: {baseline_type}, Model: {config.MODEL}, WithGroundTruth: {config.WITH_GROUND_TRUTH}, Tolerance: {tolerance}, Total: {all_total}, Success: {all_success}, Avg_Accuracy: {all_success/all_total:.4f}, Avg_Tokens: {avg_tokens}\n"

        summary_file = os.path.join(output_dir, "overall_accuracy_log.txt")
        existing = open(summary_file, encoding="utf-8").read() if os.path.exists(summary_file) else ""
        if log_line not in existing:
            with open(summary_file, "a", encoding="utf-8") as f:
                f.write(log_line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fault localization evaluation")
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        choices=["all_at_once", "step_by_step", "binary_search", "agent_locate", "trail", "FAMAS"],
        help="Baseline method to evaluate"
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only score existing results in output/ without calling any API"
    )
    args = parser.parse_args()
    evaluate_all_cases(args.baseline, eval_only=args.eval_only)