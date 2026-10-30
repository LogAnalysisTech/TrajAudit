# TrajAudit: Automated Failure Diagnosis for Agentic Systems in Software Maintenance 

RootSE is also available on HuggingFace: [dengdan1999/RootSE](https://huggingface.co/datasets/dengdan1999/RootSE)

# Workflow of TrajAudit
![Workflow of TrajAudit](./resources/workflow.png)
TrajAudit consists of one investigator agent supported by two modules. The project structure follows this modular design.

# Project Structure
```
agent_system_log_normalization
├── RootSE/                          # RootSE benchmark: agent trajectories and failure annotations
│   ├── auto-code-rover/             #   AutoCodeRover agent trajectories
│   ├── openhands/                   #   OpenHands agent trajectories
│   ├── swe-agent/                   #   SWE-agent trajectories (Claude, Gemini, GPT backbones)
│   └── swe-agent-live/              #   SWE-agent-live trajectories
│
├── WhoAndWhen/                      # Who&When benchmark (Zhang et al., ICML 2025), copied from its official repo
│   ├── Algorithm-Generated/         #   126 failure logs from CaptainAgent-generated systems
│   └── Hand-Crafted/                #   58 failure logs from Magentic-One
│
├── TrajAudit/                       # Core TrajAudit implementation
│   ├── fault_localization.py        #   Base investigator agent
│   ├── fault_localization_claude.py #   Claude-specific variant (strict JSON output)
│   ├── fault_localization_generic.py#   Generic OpenAI-compatible variant
│   ├── fault_localization_who_and_when.py # Self-contained TrajAudit adaptation for Who&When
│   └── template.yaml                #   Prompt templates and tool definitions
│
├── baseline/                        # Baseline methods for comparison
│   ├── who_and_when/                #   Prompt-only baselines
│   │   ├── all_at_once.py           #     Single-pass full-trajectory prompting
│   │   ├── binary_search.py         #     Binary search over trajectory steps
│   │   └── step_by_step.py          #     Sequential per-step analysis
│   ├── trail/
│   │   └── trail_adapted.py         #   TRAIL adapted for failure localization
│   └── FAMAS/
│       ├── famas.py                 #   FAMAS adapted for failure localization
│       └── collect_*.py             #   Reference trajectory collection scripts
│
├── output/                          # Evaluation outputs
│   ├── agent_locate/                #   TrajAudit results (per backbone model)
│   │   ├── claude-sonnet-4-5-*/     #     Claude Sonnet results
│   │   ├── deepseek-v4-pro/         #     DeepSeek results
│   │   ├── gemini-3.1-pro-preview/  #     Gemini results
│   │   ├── gpt-5.2-2025-12-11/      #     GPT results
│   │   └── qwen3-235b-*/            #     Qwen results
│   ├── FAMAS/                       #   FAMAS baseline results
│   ├── TRAIL/                       #   TRAIL baseline results
│   ├── all_at_once/                 #   All-at-Once baseline results
│   ├── binary_search/               #   Binary Search baseline results
│   ├── step_by_step/                #   Step-by-Step baseline results
│   ├── who_and_when/                #   TrajAudit results on Who&When
│   │   ├── agent_locate/<model>/<subset>/   # Per-instance .traj, result.csv and eval.json
│   │   └── overall_accuracy_log.txt #     Aggregated Who&When results
│   └── overall_accuracy_log.txt     #   Aggregated evaluation results across all methods
│
├── config.py                        # API key, model, and experiment configuration
├── eval.py                          # Main evaluation entry point
├── utils.py                         # Shared utilities (SSF, data loading, etc.)
├── evaluate                         # CLI wrapper for running evaluations
├── requirements.txt                 # Python dependencies
├── resources/                       # Images used in README
└── README.md
```

# Quick Start
### 1. Environment Setup
```bash
conda create -n trajaudit python=3.12.7
conda activate trajaudit

pip install -r requirements.txt
```

### 2. Configure API credentials
Fill in your API key, base URL, and model in `config.py`:
```python
BASE_URL = "YOUR_BASE_URL"
API_KEY  = "YOUR_API_KEY"
MODEL    = "claude-sonnet-4-5-20250929"   # or any OpenAI-compatible model
```

### 3. Run evaluation
```bash
./evaluate agent_locate      # TrajAudit (ours)
./evaluate all_at_once
./evaluate step_by_step
./evaluate binary_search
./evaluate trail
./evaluate FAMAS
```

# Evaluation on Who&When
`TrajAudit/fault_localization_who_and_when.py` applies TrajAudit to the [Who&When](https://github.com/mingyin1/Agents_Failure_Attribution) benchmark of multi-agent failure logs. It keeps the investigator agent, Prior Failure Reasoning (PFR) and Semantic Saliency Folding (SSF), and adapts them to multi-agent conversations:
- **Input**: the user question and the conversation history. With reference, the ground-truth answer of the question is also given, as in the original Who&When prompts.
- **Annotation guideline**: the Who&When definition, i.e., the agent responsible for the wrong solution and the step where it first made the decisive error.
- **PFR**: infers the failure stage from the final steps of the conversation (and the correct answer, with reference), using a multi-agent counterpart of the RootSE stage taxonomy.
- **SSF**: folds long outputs of tool agents (WebSurfer, FileSurfer, ComputerTerminal, Computer_terminal) unless they contain failure keywords.
- **Output**: the decisive error step and the responsible agent, scored separately.

### 1. Configure
Set `BASE_URL`, `API_KEY` and `MODEL` in `config.py` as above. The flags below are shared with RootSE:
```python
WITH_GROUND_TRUTH = True               # give the ground-truth answer of the question
WITH_DIAGNOSIS = True                  # Prior Failure Reasoning
WITH_CONTEXT_AWARE_TRUNCATION = True   # Semantic Saliency Folding
```

### 2. Run and evaluate
Run from the repository root:
```bash
# all 184 instances (both subsets), then evaluate
python -m TrajAudit.fault_localization_who_and_when --subset all

# one subset: hand_crafted or algorithm_generated
python -m TrajAudit.fault_localization_who_and_when --subset hand_crafted

# the first N instances of each subset
python -m TrajAudit.fault_localization_who_and_when --subset all --limit 5

# re-evaluate existing results without calling the LLM
python -m TrajAudit.fault_localization_who_and_when --subset all --eval_only
```
Other options: `--workers` (parallel instances, default 8) and `--data_dir` (default `./WhoAndWhen`). Finished instances are skipped, so an interrupted run can be resumed with the same command.

Results are written to `output/who_and_when/agent_locate/<model>/<subset>/<setting>/runtime_<k>/`:
- `<instance_id>.traj`: the investigator's interaction history
- `result.csv`: predicted step, agent, explanation, PFR stage and token usage per instance
- `eval.json`: predictions next to the annotations

Agent-level and step-level accuracy (Tol@0 to Tol@3) are printed for each subset and for both subsets combined, and appended to `output/who_and_when/overall_accuracy_log.txt`. Matching is strict: the predicted step must equal the annotated step (Tol@k allows a difference of up to k steps), and the predicted agent must equal the annotated agent.