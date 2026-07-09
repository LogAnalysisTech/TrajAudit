"""
FAMAS: Failure Attribution for Multi-Agent Systems via Spectrum Analysis
Adapted for the RootSE benchmark (single-agent SWE trajectories).

Reference: Ge et al., "Who is Introducing the Failure? Automatically Attributing
           Failures of Multi-Agent Systems via Spectrum Analysis", arXiv:2509.13782

Two-phase pipeline:
  Phase 1 — Trajectory Abstraction
    1.1  Deterministic extraction: each step → "action_type:primary_param" key
         (e.g. search_method:get_form_kwargs, editor_view:widgets.py, grep:db_table)
    1.2  LLM hierarchical clustering: semantically equivalent keys across all
         trajectories for this instance are merged into one cluster representative
         (mirrors the paper's LLM-based clustering step).
  Phase 2 — Spectrum Analysis: compute the FAMAS suspiciousness score S(η_j) for
             each cluster representative using the λ-decay Kulczynski2 formula,
             augmented by Action Behavior Group (α).
             Note: the Agent Behavior Group (γ, β) from Eq. 6 is intentionally
             omitted for single-agent use — see _compute_scores() docstring.
             The step in τ_0 whose cluster has the highest score is returned as
             failure_id.

If no reference trajectories are found for the instance → failure_id = -1.

Reference trajectory format (each JSON file under
  baseline/FAMAS/reference_trajs/{agent}/{benchmark}/{instance_id}/):
{
    "passed": true | false,
    "steps": [
        {"action": "...", "observation": "..."},
        ...
    ]
}
Steps may also be given in the full RootSE format (with "thought", "index", etc.).
"""

import json
import math
import os
import re
import threading

import pandas as pd
from openai import OpenAI

import config
from utils import Failure, annotation_guidelines

csv_lock = threading.Lock()

LAMBDA = 0.9   # λ decay factor (optimal per paper ablation study)

# Use the same model as config for the LLM clustering step (Phase 1.2).
# Set to None (or a specific model name) to override.
CLUSTER_MODEL = None  # None → use config.MODEL

# Maximum number of reference trajectories to use per instance.
# Selection keeps a balanced pass/fail ratio and prefers longer trajectories.
MAX_REF_TRAJS = 10


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze_famas(failure: Failure, output_dir: str, runtime: int):
    FAMASAnalyzer(failure, output_dir, runtime).run()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FAMASAnalyzer:
    def __init__(self, failure: Failure, output_dir: str, runtime: int):
        folder_path = os.path.join(
            output_dir, "FAMAS", config.MODEL,
            f"GROUND_TRUTH_{config.WITH_GROUND_TRUTH}",
            f"runtime_{runtime}",
        )
        self.filepath = os.path.join(folder_path, "result.csv")
        with csv_lock:
            os.makedirs(folder_path, exist_ok=True)

        self.failure = failure
        self.lambda_ = LAMBDA

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def run(self):
        print(f"[FAMAS] {self.failure.instance_id}")

        # Load reference trajectories
        ref_trajs, outcomes = self._load_reference_trajs()
        has_passing_refs = sum(outcomes) > 0 if outcomes else False

        if not ref_trajs:
            # No refs at all: LLM-only prediction on τ_0
            all_trajs = [self.failure.original_traj]
            traj_action_keys, traj_thoughts, extract_tokens = \
                self._extract_action_keys_llm_batch(all_trajs)
            failure_step, reason, llm_tokens = self._llm_final_prediction(
                tau0_steps=self.failure.original_traj,
                tau0_keys=traj_action_keys[0],
                tau0_thoughts=traj_thoughts[0],
                ranked_clusters=[],
                has_famas_signal=False,
            )
            total_tokens = extract_tokens + llm_tokens
            print(f"[FAMAS]   → no refs, LLM-only: step={failure_step}")
            self._save_result(str(failure_step), reason, tokens=total_tokens)
            return

        # τ_0 = failing trajectory (index 0), then reference trajectories
        all_trajs = [self.failure.original_traj] + ref_trajs
        all_outcomes = [0] + outcomes   # 0 = fail, 1 = pass

        print(f"[FAMAS]   Loaded {len(ref_trajs)} reference trajectories "
              f"({sum(outcomes)} passing, {len(outcomes) - sum(outcomes)} failing)")

        # ------------------------------------------------------------------
        # Phase 1.1: LLM semantic action extraction
        #   For every step across all trajectories, ask the LLM to produce
        #   a "verb:object" description (e.g. "view:models.py",
        #   "search:get_form_kwargs", "run_test:test_forms.py").
        #   The |state suffix is appended deterministically from the observation.
        # ------------------------------------------------------------------
        traj_action_keys, traj_thoughts, extract_tokens = \
            self._extract_action_keys_llm_batch(all_trajs)

        all_unique_keys = sorted({k for keys in traj_action_keys for k in keys})
        print(f"[FAMAS]   {len(all_unique_keys)} unique semantic keys extracted "
              f"(tokens: {extract_tokens})")

        # ------------------------------------------------------------------
        # Phase 1.2: LLM hierarchical clustering
        # ------------------------------------------------------------------
        cluster_map, cluster_tokens = self._cluster_action_keys(all_unique_keys)
        unique_reps = sorted(set(cluster_map.values()))
        print(f"[FAMAS]   → {len(unique_reps)} cluster representatives "
              f"(cluster_tokens: {cluster_tokens})")

        traj_clustered = [
            [cluster_map.get(k, k) for k in keys]
            for keys in traj_action_keys
        ]

        # ------------------------------------------------------------------
        # Phase 2: Spectrum analysis (meaningful only when has_passing_refs)
        #   Produces a ranked list of suspicious clusters; even when refs are
        #   all-failing the ranking gives the LLM useful structural context.
        # ------------------------------------------------------------------
        U_L = sorted({k for keys in traj_clustered for k in keys})
        triple_to_idx = {t: j for j, t in enumerate(U_L)}
        k_plus_1 = len(all_trajs)
        m = len(U_L)

        C = [[0] * m for _ in range(k_plus_1)]
        F = [[0] * m for _ in range(k_plus_1)]
        for i, keys in enumerate(traj_clustered):
            for k in keys:
                j = triple_to_idx[k]
                C[i][j] = 1
                F[i][j] += 1

        O = all_outcomes
        scores = self._compute_scores(C, F, O, U_L, triple_to_idx, k_plus_1)

        # Build ranked top-10 clusters that appear in τ_0, with their step positions
        tau0_clustered = traj_clustered[0]
        tau0_cluster_set = set(tau0_clustered)
        ranked_clusters = sorted(
            [(c, s) for c, s in scores.items() if c in tau0_cluster_set],
            key=lambda x: x[1], reverse=True
        )[:10]
        ranked_with_steps = [
            (cluster, score,
             [i for i, k in enumerate(tau0_clustered) if k == cluster])
            for cluster, score in ranked_clusters
        ]

        print(f"[FAMAS]   Top cluster: '{ranked_clusters[0][0] if ranked_clusters else 'none'}' "
              f"S={ranked_clusters[0][1]:.4f}" if ranked_clusters else "[FAMAS]   No scored clusters")

        # ------------------------------------------------------------------
        # Phase 3: LLM final prediction
        #   Give LLM the compact τ_0 trajectory (extracted keys) + FAMAS
        #   ranking + annotation guideline → single step prediction.
        #   For zero-passing-ref cases the FAMAS ranking is still shown but
        #   the LLM is told it is less reliable.
        # ------------------------------------------------------------------
        failure_step, reason, llm_tokens = self._llm_final_prediction(
            tau0_steps=self.failure.original_traj,
            tau0_keys=traj_action_keys[0],
            tau0_thoughts=traj_thoughts[0],
            ranked_clusters=ranked_with_steps,
            has_famas_signal=has_passing_refs,
        )

        total_tokens = extract_tokens + cluster_tokens + llm_tokens
        print(f"[FAMAS]   → LLM final: step={failure_step}  total_tokens={total_tokens}")
        self._save_result(str(failure_step), reason, tokens=total_tokens)

    # ------------------------------------------------------------------
    # Phase 2: spectrum score computation
    # ------------------------------------------------------------------

    def _compute_scores(self, C, F, O, U_L, triple_to_idx, k_plus_1):
        """Compute S(η_j) for each triple that appears in τ_0.

        Single-agent adaptation note
        ─────────────────────────────
        The original FAMAS formula (Eq. 6) is:

            S(η_j) = α · Kul2^λ(η_j) · (1 + β) · (1 + γ)

        where (1+β)(1+γ) is the Agent Behavior Group (ABG), designed to
        *rank agents* in a multi-agent pipeline by how much each agent's
        actions contribute to the failure.

        In our single-agent setting every trajectory belongs to the same
        agent, so:
          • γ = nc_η_j / nc_agent = nc_η_j / (k+1)
            Common exploration actions (editor_view|output, read_file|output,
            etc.) appear in nearly every trajectory → γ ≈ 1.0 → (1+γ) ≈ 2.0
          • This amplifies ubiquitous early-step actions and severely biases
            the predicted failure step toward step 1–3, where those common
            actions first appear.  Empirically this produced mean predicted
            step = 4.9 vs. mean ground-truth step = 18.8.

        Fix: drop the ABG factors for single-agent use.
        The score reduces to S = α · Kul2^λ, which is the core SBFL signal:
          • α rewards actions that are *more frequent* in the failing run
          • Kul2^λ rewards actions that are *specific to failing* trajectories

        This is equivalent to the λ-decay SBFL baseline from the paper
        (Section 4.1) with the Local Frequency Enhancement (α) factor, which
        is well-suited for single-agent step localization.
        """
        scores = {}

        for j, triple in enumerate(U_L):
            # Only score triples that appear in the original failing trajectory (τ_0)
            if C[0][j] == 0:
                continue

            # ---- Action Behavior Group (α) ----
            # α: Local Frequency Enhancement for τ_0
            # α_{τ_0}(η_j) = 1 + log_{1/λ}(f_{0j})
            # log_{1/λ}(x) = ln(x) / ln(1/λ) = -ln(x)/ln(λ)
            f0j = F[0][j]
            alpha = 1.0 + (math.log(f0j) / math.log(1.0 / self.lambda_)) if f0j > 1 else 1.0

            # λ-Decay SBFL coefficients
            n_cf_lambda = 0.0   # weighted failing coverage
            n_cs_lambda = 0.0   # weighted passing coverage
            for i in range(k_plus_1):
                fpj = F[i][j]
                if fpj > 0:
                    decay = self.lambda_ ** (fpj - 1)
                    if O[i] == 0:
                        n_cf_lambda += decay
                    else:
                        n_cs_lambda += decay

            # n_uf: plain count of failing trajectories NOT covering triple j
            n_uf = sum(1 for i in range(k_plus_1) if O[i] == 0 and C[i][j] == 0)

            # λ-decay Kulczynski2
            denom1 = n_cf_lambda + n_uf
            denom2 = n_cf_lambda + n_cs_lambda
            if denom1 <= 0 and denom2 <= 0:
                kul2 = 0.0
            elif denom1 <= 0:
                kul2 = 0.5 * (0.0 + n_cf_lambda / denom2)
            elif denom2 <= 0:
                kul2 = 0.5 * (n_cf_lambda / denom1 + 0.0)
            else:
                kul2 = 0.5 * (n_cf_lambda / denom1 + n_cf_lambda / denom2)

            # ---- Final score (S = α · Kul2^λ, matching paper Eq.6 base) ----
            S = alpha * kul2
            scores[triple] = S

        return scores

    # ------------------------------------------------------------------
    # Phase 3: LLM final prediction
    #   Receives the compact τ_0 representation (extracted semantic keys)
    #   and the FAMAS-ranked suspicious clusters, then asks the LLM to
    #   identify the failure-introducing step.
    # ------------------------------------------------------------------

    def _llm_final_prediction(
        self,
        tau0_steps: list,
        tau0_keys: list,
        tau0_thoughts: list,
        ranked_clusters: list,
        has_famas_signal: bool,
    ) -> tuple:
        """
        Phase 3: Guided LLM final prediction.

        Combines the full RootSE annotation guideline used by all other
        baselines with FAMAS spectrum ranking and a rich per-step trajectory
        representation (action key + thought summary + observation snippet).

        Ground truth (gold_patch) is included when config.WITH_GROUND_TRUTH=True,
        consistent with how all_at_once/trail/agent_locate handle it.

        Returns (failure_step: int, failure_reason: str, tokens_used: int).
        """
        # ---- per-step trajectory representation --------------------------
        # For each τ_0 step: action key + thought (if any) + obs snippet.
        # FAMAS rank is annotated inline for top-10 clusters.
        step_rank = {}  # step_idx → rank label
        for r, (cluster, score, steps) in enumerate(ranked_clusters):
            for s in steps:
                if s not in step_rank:
                    step_rank[s] = f"[FAMAS #{r+1} S={score:.2f}]"

        traj_lines = []
        for i, step_item in enumerate(tau0_steps):
            step = json.loads(step_item) if isinstance(step_item, str) else step_item
            key     = tau0_keys[i] if i < len(tau0_keys) else "unknown"
            thought = tau0_thoughts[i] if tau0_thoughts and i < len(tau0_thoughts) else None

            # Observation snippet: truncate intelligently by state
            obs_raw = step.get("observation", "")
            if not isinstance(obs_raw, str):
                obs_raw = str(obs_raw)
            obs_raw = obs_raw.strip()
            state = key.split("|")[-1] if "|" in key else "output"
            if state in ("test_failed", "error"):
                obs_snippet = obs_raw[:300].replace("\n", " ")
            elif state in ("found", "not_found"):
                obs_snippet = obs_raw[:150].replace("\n", " ")
            else:
                obs_snippet = obs_raw[:80].replace("\n", " ")

            rank_tag = f"  {step_rank[i]}" if i in step_rank else ""
            line = f"Step {i:>3}:{rank_tag}\n"
            line += f"  action : {key}\n"
            if thought:
                line += f"  thought: {thought}\n"
            if obs_snippet:
                line += f"  obs    : {obs_snippet}\n"
            traj_lines.append(line)

        traj_text = "\n".join(traj_lines)

        # ---- FAMAS ranking block -----------------------------------------
        if ranked_clusters:
            reliability = (
                "based on passing & failing reference trajectories — statistically meaningful"
                if has_famas_signal else
                "NOTE: all reference trajectories are also failing — scores are less "
                "reliable since there is no passing reference to contrast against"
            )
            rank_lines = "\n".join(
                f"  #{r+1}  '{cluster}'  score={score:.3f}  at steps {steps}"
                for r, (cluster, score, steps) in enumerate(ranked_clusters)
            )
            famas_block = (
                f"FAMAS Spectrum Analysis ({reliability}):\n"
                f"{rank_lines}\n\n"
                "Use this ranking as a statistical prior alongside your own reasoning. "
                "Higher score = more correlated with failure across reference runs.\n"
            )
        else:
            famas_block = "FAMAS Spectrum Analysis: unavailable (no reference trajectories).\n"

        # ---- ground truth block (conditional) ----------------------------
        if config.WITH_GROUND_TRUTH:
            gt_block = (
                "\nThe following is a reference implementation. Note that multiple "
                "paths may achieve the task; differing from the gold patch does not "
                "necessarily mean the agent has deviated from a correct implementation.\n"
                f"<gold_patch>\n{self.failure.gold_patch}\n</gold_patch>\n"
            )
        else:
            gt_block = ""

        # ---- full prompt (mirrors all_at_once structure) -----------------
        system_prompt = (
            "You are a helpful assistant in diagnosing and localising failures "
            "in agent systems for software engineering."
        )

        user_prompt = f"""The agent system was given the following problem to solve:

<problem_description>
{self.failure.original_mission_prompt}
</problem_description>

The system's execution trajectory is shown below (each step includes the \
semantic action, thought reasoning if available, and a key observation snippet):

<trajectory>
{traj_text}
</trajectory>

However, the task failed to pass the predefined test.

<test_code>
{self.failure.failed_test_code}
</test_code>

<error_message>
{self.failure.failed_test_error_description}
</error_message>
{gt_block}
{famas_block}
**Annotation guidelines:**
{annotation_guidelines}

**Response format requirements:**
You MUST respond in JSON format with exactly two keys:
{{
  "failure_id": "the index of the first deviation step (0-based integer as string)",
  "failure_reason": "a detailed explanation of why this specific step caused the task to fail"
}}

Combine the FAMAS spectrum ranking (as a statistical prior) with the annotation \
guidelines and your own reading of the trajectory to determine the most likely \
failure-introducing step.

- Do NOT include markdown code blocks
- Do NOT include any text before or after the JSON object
"""

        try:
            client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)
            response = client.chat.completions.create(
                model=config.MODEL,
                max_tokens=1024,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            tokens = response.usage.total_tokens if response.usage else 0
            text   = response.choices[0].message.content or ""

            data = json.loads(text)
            fid  = int(data["failure_id"])
            fid  = max(0, min(fid, len(tau0_steps) - 1))
            return fid, data.get("failure_reason", ""), tokens

        except Exception as e:
            print(f"[FAMAS] LLM final prediction failed: {e}")
            # Fallback: try extracting just the integer
            m = re.search(r'"failure_id"\s*:\s*"?(\d+)"?', text if 'text' in dir() else "")
            if m:
                fid = max(0, min(int(m.group(1)), len(tau0_steps) - 1))
                return fid, f"partial parse: {e}", tokens if 'tokens' in dir() else 0

        return -1, f"LLM prediction failed", 0

    # ------------------------------------------------------------------
    # Phase 1.1 (LLM): Semantic action extraction — batch all trajectories
    # ------------------------------------------------------------------

    def _format_step_for_llm(self, step: dict) -> tuple:
        """
        Return (action_str, thought_str_or_none) for a step.
        action_str  — compact human-readable action (max 200 chars)
        thought_str — raw thought/reasoning text (max 300 chars), or None
        """
        # ---- thought field --------------------------------------------------
        thought_raw = step.get("thought", step.get("thinking", ""))
        thought = thought_raw.strip()[:300] if thought_raw and thought_raw.strip() else None

        # ---- action string --------------------------------------------------
        action = step.get("action", "")
        if isinstance(action, list):
            parts = []
            for item in action:
                if isinstance(item, dict) and "func_name" in item:
                    args = item.get("arguments", {})
                    arg_str = ", ".join(f"{k}={str(v)[:40]}" for k, v in args.items())
                    parts.append(f"{item['func_name']}({arg_str})")
            action_str = "; ".join(parts)[:200] if parts else "unknown"
        elif isinstance(action, dict):
            tool = action.get("tool", action.get("type", ""))
            inp  = action.get("input", {})
            if tool == "str_replace_editor":
                subcmd = inp.get("command", "view")
                path   = inp.get("path", "")
                action_str = f"str_replace_editor {subcmd} {path}"[:200]
            elif tool == "execute_bash":
                action_str = inp.get("command", "")[:200]
            else:
                first_val = str(next(iter(inp.values()), ""))[:80] if inp else ""
                action_str = f"{tool}({first_val})"[:200]
        else:
            action_str = str(action).strip()[:200]

        return action_str, thought

    # kept for backward-compat (used in fallback path)
    def _format_action_for_llm(self, step: dict) -> str:
        return self._format_step_for_llm(step)[0]

    def _extract_action_keys_llm_batch(self, all_trajs: list) -> tuple:
        """
        Phase 1.1 (LLM): For every step across ALL trajectories, ask the LLM to
        produce a short 'verb:object' semantic description, then append |state
        from deterministic observation analysis.

        All trajectories are batched into a single LLM call (≈350 steps
        × 200-char actions ≈ 18k input tokens — well within limits).

        Returns (traj_action_keys, traj_thought_summaries, tokens_used).
          traj_action_keys     : list[list[str]] — 'verb:object|state' per step
          traj_thought_summaries: list[list[str|None]] — brief thought summary or None
        Falls back to rule-based extraction (no thought summaries) on any error.
        """
        # ---- 1. Collect states deterministically for every step ------------
        all_states: list[list[str]] = []
        for traj in all_trajs:
            states = []
            for step_item in traj:
                step = json.loads(step_item) if isinstance(step_item, str) else step_item
                states.append(self._get_outcome_category(step))
            all_states.append(states)

        # ---- 2. Build the batch prompt --------------------------------------
        lines = []
        for ti, traj in enumerate(all_trajs):
            for si, step_item in enumerate(traj):
                step = json.loads(step_item) if isinstance(step_item, str) else step_item
                act_str, thought = self._format_step_for_llm(step)
                line = f"[T{ti}S{si}] action: {act_str}"
                if thought:
                    line += f"\n         thought: {thought}"
                lines.append(line)

        steps_block = "\n".join(lines)

        prompt = f"""You are analyzing steps from software engineering agent trajectories on a programming bug-fix task.

For each step labeled [TiSj], extract TWO fields:

1. "action" — concise "verb:object" description of what was done
   - verb: view, edit, create, search, run_test, run_script, grep, find, apply_patch, navigate, think, other
   - object: specific target (file basename, method/class name, test file, grep pattern ≤40 chars)
   - ALWAYS include both verb AND object separated by colon

2. "thought" — one short phrase (≤10 words) capturing the agent's CENTRAL REASONING at this step
   - Only fill if a "thought" field is provided for this step; otherwise use null
   - Summarise the agent's goal/hypothesis/inference (not the mechanical action)
   - Examples: "locating root cause of test failure", "verifying fix resolves assertion error"

Steps:
{steps_block}

Return ONLY valid JSON:
{{"T0S0": {{"action": "view:models.py", "thought": "understanding form model structure"}}, "T0S1": {{"action": "search:get_form_kwargs", "thought": null}}, ...}}
"""

        # ---- 3. LLM call ----------------------------------------------------
        try:
            client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)
            response = client.chat.completions.create(
                model=config.MODEL,
                max_tokens=10000,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            tokens = response.usage.total_tokens if response.usage else 0
            text   = response.choices[0].message.content or ""

            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m:
                try:
                    desc_map = json.loads(m.group())
                except json.JSONDecodeError:
                    desc_map = {}
                    for pm in re.finditer(
                        r'"(T\d+S\d+)"\s*:\s*\{[^}]*"action"\s*:\s*"([^"]+)"[^}]*\}', text
                    ):
                        thought_m = re.search(r'"thought"\s*:\s*"([^"]+)"', pm.group(0))
                        desc_map[pm.group(1)] = {
                            "action": pm.group(2),
                            "thought": thought_m.group(1) if thought_m else None,
                        }

                action_keys   = []
                thought_summs = []
                for ti, traj in enumerate(all_trajs):
                    t_keys    = []
                    t_thoughts = []
                    for si in range(len(traj)):
                        key_id = f"T{ti}S{si}"
                        entry  = desc_map.get(key_id, {})
                        state  = all_states[ti][si]

                        action_desc = (entry.get("action") or "").strip() \
                            if isinstance(entry, dict) else str(entry).strip()
                        thought_val = (entry.get("thought") or "").strip() \
                            if isinstance(entry, dict) else ""

                        if action_desc and ":" in action_desc:
                            t_keys.append(f"{action_desc}|{state}")
                        else:
                            step_item = traj[si]
                            step = json.loads(step_item) if isinstance(step_item, str) else step_item
                            t_keys.append(self._get_action_key(step))

                        t_thoughts.append(thought_val if thought_val else None)

                    action_keys.append(t_keys)
                    thought_summs.append(t_thoughts)

                n_covered = sum(1 for ti, traj in enumerate(all_trajs)
                                for si in range(len(traj)) if f"T{ti}S{si}" in desc_map)
                n_with_thought = sum(1 for v in desc_map.values()
                                     if isinstance(v, dict) and v.get("thought"))
                total = sum(len(t) for t in all_trajs)
                print(f"[FAMAS]   LLM covered {n_covered}/{total} steps, "
                      f"{n_with_thought} with thought summary")
                return action_keys, thought_summs, tokens

        except Exception as e:
            print(f"[FAMAS] LLM action extraction failed: {e}")

        # ---- 4. Full fallback ------------------------------------------------
        print(f"[FAMAS]   Falling back to rule-based action extraction")
        fallback_keys = [self._extract_action_keys(traj) for traj in all_trajs]
        fallback_thoughts = [[None] * len(t) for t in all_trajs]
        return fallback_keys, fallback_thoughts, 0

    # ------------------------------------------------------------------
    # Phase 1.1 (rule-based, kept as fallback): Fine-grained action key extraction
    #   Final key format:  "action_type:param|state"
    #   e.g. search_method:get_form_kwargs|not_found
    #        pytest:test_forms.py|test_failed
    #        editor_edit:widgets.py|success
    # ------------------------------------------------------------------

    def _extract_action_keys(self, traj_steps: list) -> list:
        """Return a list of 'action_type:param|state' keys, one per step."""
        keys = []
        for step_item in traj_steps:
            if isinstance(step_item, str):
                try:
                    step = json.loads(step_item)
                except Exception:
                    step = {"action": step_item}
            else:
                step = step_item
            keys.append(self._get_action_key(step))
        return keys

    def _get_action_key(self, step: dict) -> str:
        """
        Returns 'action_type:primary_param|state' for fine-grained spectrum analysis.

        ACTION part (action_type:param):
          auto-code-rover list  → method_name / class_name / code_str (first 50 chars)
          openhands dict        → file basename for editor ops
          swe-agent string      → file basename, grep pattern, test name, script name

        STATE part (|state):
          Coarse-grained outcome extracted from the observation field.
          Categories: found / not_found / error / test_failed / success / empty / output

        Falls back to action_type (without state) when no meaningful extraction possible.
        """
        action = step.get("action", "")
        state  = self._get_outcome_category(step)

        def _combine(action_part: str) -> str:
            return f"{action_part}|{state}"

        # ------------------------------------------------------------------
        # auto-code-rover: [{"func_name": ..., "arguments": {...}}, ...]
        # ------------------------------------------------------------------
        if isinstance(action, list):
            for item in action:
                if not isinstance(item, dict) or "func_name" not in item:
                    continue
                func = item["func_name"]
                args = item.get("arguments", {})

                if func in ("search_method", "search_method_in_file",
                            "search_method_in_class"):
                    t = args.get("method_name", "").strip()
                    return _combine(f"search_method:{t}" if t else "search_method")

                if func in ("search_class", "search_class_in_file"):
                    t = args.get("class_name", "").strip()
                    return _combine(f"search_class:{t}" if t else "search_class")

                if func in ("search_code", "search_code_in_file"):
                    code = args.get("code_str", "").strip()[:50]
                    return _combine(f"search_code:{code}" if code else "search_code")

                if func == "get_class_full_snippet":
                    t = args.get("class_name", "").strip()
                    return _combine(f"get_class:{t}" if t else "get_class")

                if func == "write_patch":
                    return _combine("write_patch")

                # Other funcs (list_tree, etc.)
                if args:
                    first_val = str(next(iter(args.values()), "")).strip()[:50]
                    if first_val:
                        return _combine(f"{func}:{first_val}")
                return _combine(func)
            return _combine("unknown")

        # ------------------------------------------------------------------
        # openhands: {"tool": ..., "input": {...}}
        # ------------------------------------------------------------------
        if isinstance(action, dict):
            tool = action.get("tool", action.get("type", "")).strip()
            inp  = action.get("input", {})

            if tool == "str_replace_editor":
                subcmd   = inp.get("command", "view")
                path     = inp.get("path", "")
                label    = self._EDITOR_SUBCMD_MAP.get(subcmd, f"editor_{subcmd}")
                filename = os.path.basename(path.rstrip("/")) if path else ""
                return _combine(f"{label}:{filename}" if filename else label)

            if tool == "execute_bash":
                cmd = inp.get("command", "").strip()
                if not cmd:
                    return _combine("execute_bash")
                base_type = self._parse_shell_cmd(cmd)
                param     = self._extract_shell_param(cmd, base_type)
                return _combine(f"{base_type}:{param}" if param else base_type)

            return _combine(tool or "unknown")

        # ------------------------------------------------------------------
        # swe-agent / swe-agent-live: bare shell command string
        # ------------------------------------------------------------------
        if isinstance(action, str):
            cmd = action.strip()
            if not cmd or cmd in ("''", '""'):
                return _combine("think")
            base_type = self._parse_shell_cmd(cmd)
            param     = self._extract_shell_param(cmd, base_type)
            return _combine(f"{base_type}:{param}" if param else base_type)

        return _combine("unknown")

    # ------------------------------------------------------------------
    # STATE extraction: outcome category from observation
    # ------------------------------------------------------------------

    def _get_outcome_category(self, step: dict) -> str:
        """
        Extract a coarse-grained outcome category from the step's observation.

        Priority (highest to lowest specificity):
          swe-agent-live  → <returncode>N</returncode> tag (most reliable)
          openhands       → "exit code N" text marker
          auto-code-rover → "Found N …" / "not found" in structured result text
          all agents      → heuristic keyword scan of observation text

        Categories returned:
          found       – search/grep returned non-empty results
          not_found   – search returned nothing / file not found
          error       – exception, traceback, or non-zero exit code
          test_failed – test suite reported failures
          success     – command completed with no errors / explicit success marker
          empty       – no observation text at all
          output      – has content but doesn't fit other categories
        """
        obs = step.get("observation", "")
        if not isinstance(obs, str):
            obs = str(obs)
        obs = obs.strip()

        if not obs:
            return "empty"

        obs_lower = obs.lower()

        # ------------------------------------------------------------------
        # 0. File-view observations (str_replace_editor / cat -n)
        #    These show raw file content; classifying based on keywords inside
        #    the file would produce false positives.  Return "output" early.
        # ------------------------------------------------------------------
        if (obs.startswith("Here's the result of running `cat -n`")
                or obs.startswith("Here's the files and directories")
                or obs_lower.startswith("here's the result of running `cat -n`")
                or obs_lower.startswith("here's the files and directories")):
            return "output"

        # ------------------------------------------------------------------
        # 1. swe-agent-live: <returncode>N</returncode>
        # ------------------------------------------------------------------
        rc_match = re.search(r'<returncode>(\d+)</returncode>', obs)
        if rc_match:
            rc = int(rc_match.group(1))
            if rc != 0:
                # Non-zero can still mean test failure – check output content
                output_match = re.search(r'<output>(.*?)</output>', obs, re.DOTALL)
                inner = output_match.group(1).strip() if output_match else ""
                if inner and any(p in inner.lower() for p in
                                 ["failed", "error", "assert"]):
                    return "test_failed"
                return "error"
            # returncode = 0
            output_match = re.search(r'<output>(.*?)</output>', obs, re.DOTALL)
            inner = (output_match.group(1).strip() if output_match else obs).lower()
            if not inner:
                return "success"
            if any(p in inner for p in ["no results", "not found", "0 results",
                                         "no such file", "no matches"]):
                return "not_found"
            return "output"

        # ------------------------------------------------------------------
        # 2. openhands: "exit code N" / "[The command completed with exit code N]"
        # ------------------------------------------------------------------
        ec_match = re.search(r'exit code (\d+)', obs_lower)
        if ec_match:
            ec = int(ec_match.group(1))
            if ec != 0:
                if any(p in obs_lower for p in ["failed", "error", "assert"]):
                    return "test_failed"
                return "error"
            # exit code 0
            if any(p in obs_lower for p in ["failed", "assertionerror",
                                             "test failed", "failures="]):
                return "test_failed"
            return "success"

        # ------------------------------------------------------------------
        # 3. auto-code-rover structured result text
        #    Note: ACR often bundles multiple search calls in a single step,
        #    so one observation can contain both "Found N classes" and
        #    "Could not find method …".  Check for any success FIRST so that
        #    mixed observations are labelled "found" rather than "not_found".
        # ------------------------------------------------------------------
        if re.search(r'result of (search_|get_)', obs_lower):
            # Any positive hit anywhere in the observation?
            if re.search(r'found \d+ (class|method|code|result|snippet)', obs_lower):
                return "found"
            if re.search(r'found 0|not found|no results|could not find', obs_lower):
                return "not_found"

        # ------------------------------------------------------------------
        # 4. Heuristic keyword scan (all agents)
        # ------------------------------------------------------------------

        # Test failures (check before generic error – overlapping keywords)
        test_fail_signals = [
            "failed", "assertionerror", "test failed", "tests failed",
            "failures=", " error in ", "e   assert",
        ]
        test_context_signals = [
            "test session", "passed", "====", "----",
            "pytest", "unittest",
        ]
        if (any(p in obs_lower for p in test_fail_signals)
                and any(p in obs_lower for p in test_context_signals)):
            return "test_failed"

        # Errors / exceptions
        error_signals = [
            "traceback (most recent call last)", "exception:",
            "syntaxerror", "typeerror", "valueerror", "attributeerror",
            "nameerror", "indentationerror", "importerror", "runtimeerror",
            "keyerror", "indexerror", "zerodivisionerror", "oserror",
            "filenotfounderror", "permissionerror",
            "error:", "errno", "killed", "segmentation fault",
            "command not found", "permission denied",
            "syntax error",      # bash: -c: line N: syntax error near ...
        ]
        if any(p in obs_lower for p in error_signals):
            return "error"

        # Not found
        not_found_signals = [
            "no results found", "not found", "could not find",
            "0 results", "no matches", "does not exist",
            "no such file", "no such directory",
        ]
        if any(p in obs_lower for p in not_found_signals):
            return "not_found"

        # Explicit grep / search output → found
        # Handles three common formats:
        #   file.py:123:content   (grep -n with filename)
        #   file.py:content       (grep without -n: path contains '/')
        #   123:content           (grep -n without filename prefix)
        if re.search(r'[\w/.-]+:\d+:', obs):          # file:lineno:content
            return "found"
        if re.search(r'(?:[\w.-]+/)+[\w.-]+:\s', obs): # /a/b/file.py: content
            return "found"
        if re.search(r'^\d+:\s', obs, re.MULTILINE):  # 123:  content (no file prefix)
            return "found"

        return "output"

    def _unwrap_cmd(self, cmd: str) -> str:
        """
        Strip bash -c / timeout / env-var wrappers recursively to return
        the innermost effective command string for parameter extraction.
        """
        cmd = cmd.strip()
        if not cmd:
            return cmd

        # Strip leading VAR=value assignments
        cmd = re.sub(r'^([A-Za-z_][A-Za-z0-9_]*=\S*\s+)+', '', cmd).strip()

        # Strip bash $'...' ANSI-C quoting prefix
        if cmd.startswith("$'") or cmd.startswith('$"'):
            cmd = cmd[2:].strip()

        tokens = cmd.split()
        if not tokens:
            return cmd
        head = tokens[0].lstrip("'\"")

        # timeout N CMD → recurse on CMD
        if head == "timeout" and len(tokens) >= 3:
            return self._unwrap_cmd(" ".join(tokens[2:]))

        # cd PATH && CMD  —  the interesting part is CMD after the &&
        if head == "cd" and "&&" in cmd:
            after_and = cmd.split("&&", 1)[1].strip()
            if after_and:
                return self._unwrap_cmd(after_and)

        # bash -lc / bash -c "CMD" → extract inner, recurse
        if head == "bash" and len(tokens) >= 2:
            c_idx = None
            for i in range(1, len(tokens)):
                tok = tokens[i]
                if not tok.startswith("-") or tok.startswith("--"):
                    break
                if "c" in tok:
                    c_idx = i
                    break
            if c_idx is not None:
                raw_inner = " ".join(tokens[c_idx + 1:]).strip()
                for q in ('"', "'"):
                    if raw_inner.startswith(q):
                        end = raw_inner.rstrip("}").rfind(q)
                        if end > 0:
                            raw_inner = raw_inner[1:end]
                            break
                inner = raw_inner.strip()
                if inner:
                    return self._unwrap_cmd(inner)

        return cmd

    def _extract_shell_param(self, cmd: str, action_type: str) -> str:
        """
        Given an already-parsed action_type, extract the most meaningful
        parameter from the shell command string.

        Automatically unwraps bash/timeout wrappers before extracting.
        Returns an empty string when no useful param can be extracted.
        """
        try:
            effective = self._unwrap_cmd(cmd)
            return self._extract_shell_param_inner(effective, action_type)
        except Exception:
            return ""

    def _extract_shell_param_inner(self, cmd: str, action_type: str) -> str:
        import shlex

        # ---- editor ops: extract file basename ---------------------------
        if action_type.startswith("editor_"):
            # str_replace_editor view /path/to/file.py → "file.py"
            parts = cmd.split()
            for part in parts[2:]:            # skip tool name + subcommand
                if part.startswith("/") or (not part.startswith("-") and "." in part):
                    return os.path.basename(part.split(":")[0].rstrip("/"))
            return ""

        # ---- grep / rg: extract search pattern ---------------------------
        if action_type == "grep":
            try:
                tokens = shlex.split(cmd)
            except Exception:
                tokens = cmd.split()
            for tok in tokens[1:]:
                if not tok.startswith("-"):
                    return tok.strip("'\"")[:50]
            return ""

        # ---- pytest: extract test module name ----------------------------
        if action_type == "pytest":
            parts = cmd.split()
            # Find "pytest" token in command (may be preceded by python3 -m)
            # and look for the test path argument after it
            pytest_idx = next(
                (i for i, p in enumerate(parts) if p == "pytest"), None
            )
            start = (pytest_idx + 1) if pytest_idx is not None else 1
            for p in parts[start:]:
                if not p.startswith("-"):
                    return os.path.basename(p.split("::")[0])
            return ""

        # ---- python script: extract script basename ----------------------
        if action_type == "python_script":
            parts = cmd.split()
            for p in parts[1:]:
                if p.endswith(".py") and not p.startswith("-"):
                    return os.path.basename(p)
            return ""

        # ---- find: extract -name pattern ---------------------------------
        if action_type == "find":
            m = re.search(r'-name\s+["\']?([^\s"\']+)', cmd)
            return m.group(1) if m else ""

        # ---- sed: extract the substitution target ------------------------
        if action_type == "sed":
            m = re.search(r's[/|,]([^/|,]+)', cmd)
            return m.group(1)[:40] if m else ""

        return ""

    # ------------------------------------------------------------------
    # Phase 1.2: LLM hierarchical clustering
    # ------------------------------------------------------------------

    def _cluster_action_keys(self, unique_keys: list) -> tuple:
        """
        Call the LLM to cluster semantically equivalent action keys into
        representatives, mirroring the paper's hierarchical clustering step.

        Returns (mapping: str→str, tokens_used: int).
        """
        if len(unique_keys) <= 1:
            return {k: k for k in unique_keys}, 0

        # When there are many unique keys the LLM response can exceed the token
        # budget.  Cap at MAX_CLUSTER_KEYS by taking the most frequent ones
        # (those that appear in multiple trajectories are the most worth merging).
        MAX_CLUSTER_KEYS = 100
        cluster_keys = list(unique_keys)
        if len(cluster_keys) > MAX_CLUSTER_KEYS:
            cluster_keys = sorted(unique_keys)[:MAX_CLUSTER_KEYS]

        keys_text = "\n".join(f"- {k}" for k in sorted(cluster_keys))

        prompt = f"""You are analyzing action traces from a software engineering agent solving the same programming task across multiple runs.

Each action key has the format "action_type:param|state", where the |state suffix captures the outcome
(e.g., |found, |not_found, |error, |test_failed, |success, |output, |empty).

Group the following action keys into semantic clusters. Two keys belong in the same cluster if they
represent the same conceptual operation AND have the same |state suffix.

Clustering rules:
1. search_method:X|s  +  search_method_in_class:X|s  +  search_method_in_file:X|s  →  same cluster
   when X is the same method name AND |s is the same state.
2. search_code:X|s  +  search_code_in_file:X|s  →  same cluster when X is the same code string AND |s is the same state.
3. search_class:X|s  +  search_class_in_file:X|s  →  same cluster when X is the same class name AND |s is the same state.
4. CRITICAL — editor_view / editor_edit / editor_create:
   - editor_view:fileA|s  and  editor_view:fileB|s  →  DIFFERENT clusters (different files = different actions).
   - NEVER merge editor_view keys with different file names. Each unique filename stays in its own cluster.
   - editor_view:file|s that appears multiple times across trajectories is already the same key — one cluster.
   - Same rule applies to editor_edit and editor_create (different files → different clusters).
5. Different search targets (different method names, class names, code strings)  →  different clusters.
6. CRITICAL: Keys with different |state suffixes (e.g., |found vs |not_found) represent DIFFERENT
   outcomes and must NEVER be merged into the same cluster, even if the action type and param match.
7. Navigation actions (cd, ls, list_tree, find, pwd) with the same |state  →  one "navigation|state" cluster.
8. Structural actions (write_patch, submit, finish, applypatch)  →  each forms its own cluster.
9. think  →  its own cluster (regardless of state).
10. pytest / python_script / python_inline / go_test / go_run with the same param AND same |state → same cluster.
    But different test targets → different clusters.

Action keys to cluster:
{keys_text}

Return ONLY valid JSON (no explanation):
{{"clusters": [{{"representative": "<concise_label>", "members": ["key1", "key2"]}}]}}

Every key must appear in exactly one cluster. The representative must include the |state suffix.
Do NOT use wildcards (*) in representative names — use the actual param value.
Keep representatives concise but informative."""

        model_to_use = CLUSTER_MODEL if CLUSTER_MODEL else config.MODEL

        def _call_llm(model_name: str):
            client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)
            return client.chat.completions.create(
                model=model_name,
                max_tokens=8000,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )

        try:
            response = _call_llm(model_to_use)
            tokens = response.usage.total_tokens if response.usage else 0
            text   = response.choices[0].message.content or ""

            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                except json.JSONDecodeError:
                    # Try salvaging partial JSON: extract complete cluster objects
                    # by finding all {"representative": ..., "members": [...]} blocks
                    data = {"clusters": []}
                    for cm in re.finditer(
                        r'\{"representative"\s*:\s*"([^"]+)"\s*,\s*"members"\s*:\s*\[([^\]]+)\]',
                        text
                    ):
                        rep = cm.group(1)
                        members_raw = cm.group(2)
                        members = re.findall(r'"([^"]+)"', members_raw)
                        data["clusters"].append({"representative": rep, "members": members})
                    if not data["clusters"]:
                        raise ValueError("Could not salvage any clusters from partial JSON")
                mapping = {}
                for cluster in data.get("clusters", []):
                    rep = cluster.get("representative", "")
                    for member in cluster.get("members", []):
                        if member in unique_keys:
                            mapping[member] = rep
                # Any key not returned by LLM (including keys over the cap) → identity fallback
                for k in unique_keys:
                    if k not in mapping:
                        mapping[k] = k
                return mapping, tokens

        except Exception as e:
            print(f"[FAMAS] Clustering LLM call failed: {e}")

        # Full fallback: identity mapping
        return {k: k for k in unique_keys}, 0

    # ------------------------------------------------------------------
    # Phase 1 (legacy): action-type-only extraction  (kept for reference)
    # ------------------------------------------------------------------

    def _extract_action_types(self, traj_steps: list) -> list:
        """Return a list of action-type strings, one per step."""
        action_types = []
        for step_item in traj_steps:
            # steps may be raw str (RootSE format) or already a dict (ref traj)
            if isinstance(step_item, str):
                try:
                    step = json.loads(step_item)
                except Exception:
                    step = {"action": step_item}
            else:
                step = step_item
            action_types.append(self._get_action_type(step))
        return action_types

    # Subcommand mappings for str_replace_editor (shared by swe-agent bare string
    # and openhands dict format)
    _EDITOR_SUBCMD_MAP = {
        "view":        "editor_view",
        "str_replace": "editor_edit",
        "create":      "editor_create",
        "insert":      "editor_insert",
        "undo_edit":   "editor_undo",
    }

    def _get_action_type(self, step: dict) -> str:
        """
        Extract a normalised action-type label from a trajectory step.

        Handles all formats present in RootSE:
          • swe-agent / swe-agent-live : action is a bare shell command string
          • openhands                  : action = {"tool": ..., "input": {...}}
          • auto-code-rover            : action = [{"func_name": ..., ...}, ...]
        """
        action = step.get("action", "")

        # ------------------------------------------------------------------
        # openhands: action is a dict {"tool": ..., "input": {...}}
        # ------------------------------------------------------------------
        if isinstance(action, dict):
            tool = action.get("tool", action.get("type", "")).strip()

            # execute_bash → drill into the inner shell command
            if tool == "execute_bash":
                cmd = action.get("input", {}).get("command", "")
                return self._parse_shell_cmd(cmd) if cmd.strip() else "execute_bash"

            # str_replace_editor → break out by subcommand (view/edit/create/insert)
            if tool == "str_replace_editor":
                subcmd = action.get("input", {}).get("command", "view")
                return self._EDITOR_SUBCMD_MAP.get(subcmd, f"editor_{subcmd}")

            # finish / task_tracker / other named tools → return tool name as-is
            return tool or "unknown"

        # ------------------------------------------------------------------
        # auto-code-rover: action is a list of {"func_name": ..., ...}
        # ------------------------------------------------------------------
        if isinstance(action, list):
            for item in action:
                if isinstance(item, dict) and "func_name" in item:
                    return item["func_name"]   # search_method, write_patch, ...
            return "unknown"

        # ------------------------------------------------------------------
        # swe-agent / swe-agent-live: action is a bare shell command string
        # Also handles openhands steps whose action is an empty string ''
        # (pure reasoning/thinking steps with no tool call)
        # ------------------------------------------------------------------
        if isinstance(action, str):
            cmd = action.strip()
            if not cmd or cmd in ("''", '""'):
                return "think"          # openhands reasoning-only step
            return self._parse_shell_cmd(cmd)

        return "unknown"

    def _parse_shell_cmd(self, cmd: str) -> str:
        """
        Parse a bare shell command string into a normalised action-type label.

        Handles all shell patterns found in RootSE swe-agent / swe-agent-live data:
          • env-var prefix      PYTHONPATH=lib python …  → strip, recurse
          • timeout prefix      timeout 60 python …      → strip, recurse
          • bash wrapper        bash -lc "CMD"            → extract CMD, recurse
          • python variants     python3 -m pytest / -c / script.py / heredoc
          • str_replace_editor  view / str_replace / create / insert / undo_edit
          • git subcommand      git diff / status / add / commit …
          • go subcommand       go build / mod / version …
          • file-reading ops    cat / head / tail / nl   → read_file
          • misc shell tools    grep / find / sed / …
        """
        import re

        cmd = cmd.strip()
        if not cmd:
            return "unknown"

        # ---- Strip leading VAR=value assignments -------------------------
        # e.g. "PYTHONPATH=lib python …"  →  "python …"
        cmd = re.sub(r'^([A-Za-z_][A-Za-z0-9_]*=\S*\s+)+', '', cmd).strip()
        if not cmd:
            return "unknown"

        # ---- Strip bash $'...' ANSI-C quoting prefix ---------------------
        # e.g. "$'grep -RIn ...'}"  →  "grep -RIn ...'"
        if cmd.startswith("$'") or cmd.startswith('$"'):
            cmd = cmd[2:].strip()
            if not cmd:
                return "unknown"

        tokens = cmd.split()
        # Strip leading quote artifact (e.g. "'str_replace" → "str_replace")
        head = tokens[0].lstrip("'\"")

        # ---- timeout N CMD  →  recurse on CMD ----------------------------
        if head == "timeout" and len(tokens) >= 3:
            return self._parse_shell_cmd(" ".join(tokens[2:]))

        # ---- cd PATH && CMD  →  real action is CMD -----------------------
        # swe-agent always uses compound form: "cd /repo && pytest tests/"
        # so the meaningful action is the part after &&, not the cd itself.
        if head == "cd" and "&&" in cmd:
            after_and = cmd.split("&&", 1)[1].strip()
            if after_and:
                return self._parse_shell_cmd(after_and)
            return "cd"

        # ---- bash -lc / bash -c "CMD"  →  extract & recurse --------------
        if head == "bash" and len(tokens) >= 2:
            # Scan only the leading bash flags (consecutive tokens starting with -)
            # to find bash's own -c flag.  Stop as soon as we hit a non-flag token
            # so that python's -c (inside the inner command) is never confused with
            # bash's -c.
            c_idx = None
            for i in range(1, len(tokens)):
                tok = tokens[i]
                if not tok.startswith("-") or tok.startswith("--"):
                    break                               # reached the command operand
                if "c" in tok:                          # -c or -lc or -ilc …
                    c_idx = i
                    break
            if c_idx is not None:
                # Re-join everything after the flag and strip outer quote pair
                raw_inner = " ".join(tokens[c_idx + 1:]).strip()
                # Remove matching outer quotes ("…", '…') accounting for trailing }
                # that appears as a data artifact in some steps
                for q in ('"', "'"):
                    if raw_inner.startswith(q):
                        end = raw_inner.rstrip("}").rfind(q)
                        if end > 0:
                            raw_inner = raw_inner[1:end]
                            break
                inner = raw_inner.strip()
                if inner:
                    return self._parse_shell_cmd(inner)
            return "bash"

        # ---- python / python3 / python2 ----------------------------------
        if head in ("python", "python3", "python2"):
            if len(tokens) >= 3 and tokens[1] == "-m":
                return tokens[2]                    # pytest, unittest, flake8 …
            if len(tokens) >= 2 and tokens[1] == "-c":
                return "python_inline"              # python3 -c "…"
            if len(tokens) >= 2 and tokens[1] in ("-", "<<", "<<'PY'", '<<"PY"'):
                return "python_inline"              # heredoc: python - << 'PY'
            if len(tokens) >= 2 and tokens[1].startswith("<<"):
                return "python_inline"
            return "python_script"                  # python script.py

        # ---- str_replace_editor (swe-agent bare-string form) -------------
        if head == "str_replace_editor":
            subcmd = tokens[1].lstrip("'\"") if len(tokens) >= 2 else "view"
            return self._EDITOR_SUBCMD_MAP.get(subcmd, f"editor_{subcmd}")

        # ---- git subcommand ----------------------------------------------
        if head == "git" and len(tokens) >= 2:
            return f"git_{tokens[1]}"

        # ---- go subcommand -----------------------------------------------
        if head == "go" and len(tokens) >= 2:
            return f"go_{tokens[1]}"

        # ---- file-reading ops  →  single label ---------------------------
        if head in ("cat", "head", "tail", "nl", "less", "more"):
            return "read_file"

        # ---- Ctrl-C / interrupt signal  ----------------------------------
        if head in ("C-c", "\x03"):
            return "interrupt"

        # ---- Well-known shell commands (direct mapping) ------------------
        _KNOWN = {
            "grep":       "grep",
            "rg":         "grep",    # ripgrep → same semantic as grep
            "find":       "find",
            "sed":        "sed",
            "awk":        "awk",
            "wc":         "wc",
            "ls":         "ls",
            "cd":         "cd",
            "rm":         "rm",
            "cp":         "cp",
            "mv":         "mv",
            "mkdir":      "mkdir",
            "chmod":      "chmod",
            "echo":       "echo",
            "export":     "export",
            "pwd":        "pwd",
            "perl":       "perl",
            "pip":        "pip",
            "pip3":       "pip",
            "which":      "which",
            "sudo":       "sudo",
            "kill":       "kill",
            "exec":       "exec",
            "gofmt":      "gofmt",
            "pytest":     "pytest",
            "submit":     "submit",        # swe-agent final submission
            "applypatch": "applypatch",    # git-am style patch apply
        }
        if head in _KNOWN:
            return _KNOWN[head]

        # ---- Absolute-path binary (e.g. /app/navidrome) ------------------
        if head.startswith("/"):
            return "binary_exec"

        # ---- Fallback: return head as-is (handles go, gofmt, etc.) -------
        return head

    # ------------------------------------------------------------------
    # Reference trajectory loading
    # ------------------------------------------------------------------

    def _load_reference_trajs(self):
        """
        Load pre-collected reference trajectories from:
          baseline/FAMAS/reference_trajs/{agent}/{benchmark}/{instance_id}/

        Each file should be JSON with keys:
          {
            "passed": true | false,
            "steps": [ {"action": ..., "observation": ...}, ... ]
          }

        Returns (trajs, outcomes) where trajs is list[list[str]] and
        outcomes is list[int] (1=pass, 0=fail).
        """
        agent = self.failure.agent
        instance_id = self.failure.instance_id

        # Prefer pre-built selected_refs/ (≤ MAX_REF_TRAJS per instance);
        # fall back to the full reference_trajs/ directory.
        _here = os.path.dirname(__file__)
        ref_base = os.path.join(_here, "selected_refs", agent)
        if not os.path.isdir(ref_base):
            ref_base = os.path.join(_here, "reference_trajs", agent)
        if not os.path.isdir(ref_base):
            return [], []

        # Search all benchmark sub-folders for a matching instance folder
        instance_folder = None
        for bench_name in sorted(os.listdir(ref_base)):
            bench_path = os.path.join(ref_base, bench_name)
            if not os.path.isdir(bench_path):
                continue
            candidate = os.path.join(bench_path, instance_id)
            if os.path.isdir(candidate):
                instance_folder = candidate
                break

        if not instance_folder:
            return [], []

        trajs, outcomes = [], []

        # New layout: instance_folder/success/*.json  and  instance_folder/fail/*.json
        # Legacy layout: instance_folder/*.json  (passed field read from file)
        success_dir = os.path.join(instance_folder, "success")
        fail_dir    = os.path.join(instance_folder, "fail")

        def _load_dir(dirpath: str, passed: bool):
            if not os.path.isdir(dirpath):
                return
            for fname in sorted(os.listdir(dirpath)):
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath, encoding="utf-8") as fh:
                        data = json.load(fh)
                    steps = data.get("steps", data.get("traj", data.get("original_traj", [])))
                    step_strs = [
                        json.dumps(s, indent=2, ensure_ascii=False) if isinstance(s, dict) else str(s)
                        for s in steps
                    ]
                    trajs.append(step_strs)
                    outcomes.append(1 if passed else 0)
                except Exception as e:
                    print(f"[FAMAS] Warning: could not load {fpath}: {e}")

        if os.path.isdir(success_dir) or os.path.isdir(fail_dir):
            # New success/fail subfolder layout
            _load_dir(success_dir, passed=True)
            _load_dir(fail_dir,    passed=False)
        else:
            # Legacy: JSON files directly in instance_folder, read 'passed' from file
            for fname in sorted(os.listdir(instance_folder)):
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(instance_folder, fname)
                try:
                    with open(fpath, encoding="utf-8") as fh:
                        data = json.load(fh)
                    passed = bool(data.get("passed", data.get("success", False)))
                    steps = data.get("steps", data.get("traj", data.get("original_traj", [])))
                    step_strs = [
                        json.dumps(s, indent=2, ensure_ascii=False) if isinstance(s, dict) else str(s)
                        for s in steps
                    ]
                    trajs.append(step_strs)
                    outcomes.append(1 if passed else 0)
                except Exception as e:
                    print(f"[FAMAS] Warning: could not load {fpath}: {e}")

        # Apply selection: keep at most MAX_REF_TRAJS with balanced pass/fail
        trajs, outcomes = self._select_reference_trajs(trajs, outcomes, MAX_REF_TRAJS)
        return trajs, outcomes

    @staticmethod
    def _select_reference_trajs(
        trajs: list, outcomes: list, max_n: int
    ) -> tuple:
        """
        Select up to max_n reference trajectories with a balanced pass/fail ratio.

        Strategy:
          1. Separate into passing (outcome=1) and failing (outcome=0) groups.
          2. Sort each group by trajectory length descending — longer trajectories
             have more diverse action coverage and are more informative.
          3. Allocate slots: target ceil(max_n/2) for the minority group first,
             then fill remaining slots from the majority group.

        This ensures both outcome types are represented whenever possible.
        """
        if len(trajs) <= max_n:
            return trajs, outcomes   # already within budget, nothing to do

        # Pair up and separate by outcome
        paired = list(zip(trajs, outcomes))
        pass_group = [(t, o) for t, o in paired if o == 1]
        fail_group = [(t, o) for t, o in paired if o == 0]

        # Sort each group: longest trajectory first (descending)
        pass_group.sort(key=lambda x: len(x[0]), reverse=True)
        fail_group.sort(key=lambda x: len(x[0]), reverse=True)

        n_pass_avail = len(pass_group)
        n_fail_avail = len(fail_group)

        # Balanced allocation: try to take ceil(max_n/2) from the smaller group
        n_minority_target = math.ceil(max_n / 2)          # 3 for max_n=5
        n_pass_take = min(n_pass_avail, n_minority_target)
        n_fail_take = min(n_fail_avail, max_n - n_pass_take)
        # Give unused pass slots to fail (and vice versa)
        n_pass_take = min(n_pass_avail, max_n - n_fail_take)

        selected = pass_group[:n_pass_take] + fail_group[:n_fail_take]
        sel_trajs    = [t for t, _ in selected]
        sel_outcomes = [o for _, o in selected]
        return sel_trajs, sel_outcomes

    # ------------------------------------------------------------------
    # Result persistence
    # ------------------------------------------------------------------

    def _save_result(self, failure_id: str, reason: str, tokens: int = 0):
        new_data = pd.DataFrame([{
            "agent": self.failure.agent,
            "model": self.failure.model,
            "instance_id": (
                f"{self.failure.agent}_{self.failure.model}_{self.failure.instance_id}"
            ),
            "failure_id": failure_id,
            "failure_reason": reason,
            "tokens": tokens,  # LLM tokens used for Phase 1.2 clustering
        }])
        with csv_lock:
            if os.path.exists(self.filepath):
                new_data.to_csv(self.filepath, mode="a", header=False, index=False)
            else:
                new_data.to_csv(self.filepath, index=False)
