"""
Zero-shot baseline evaluation: GPT-4o-mini, Claude Haiku, Gemini 1.5 Flash.
Logs Precision, Recall, F1 per model per framework to MLflow.
"""
import argparse, json, os, time
from pathlib import Path
import mlflow

SYSTEM_PROMPT = {
    "cypress": (
        "You are an expert QA engineer. Generate a Cypress test script from the given "
        "Jira user story. Use cy.* API only. Include describe/it blocks. "
        "Return only the JavaScript code, no explanation."
    ),
    "playwright": (
        "You are an expert QA engineer. Generate a Playwright test script from the given "
        "Jira user story. Use async/await page.* API with @playwright/test. "
        "Return only the JavaScript code, no explanation."
    ),
}

USER_TEMPLATE = """User Story: {story}

Acceptance Criteria:
{criteria}

Generate the test script:"""


def _load_jsonl(path):
    import jsonlines
    with jsonlines.open(path) as r:
        return list(r)


def _call_openai(story, criteria, framework, model="gpt-4o-mini"):
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT[framework]},
            {"role": "user",   "content": USER_TEMPLATE.format(story=story, criteria=criteria)},
        ],
        temperature=0.1, max_tokens=800,
    )
    return resp.choices[0].message.content.strip()


def _call_anthropic(story, criteria, framework, model="claude-haiku-4-5"):
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=800,
        system=SYSTEM_PROMPT[framework],
        messages=[{"role": "user",
                   "content": USER_TEMPLATE.format(story=story, criteria=criteria)}],
    )
    return resp.content[0].text.strip()


def _call_mlx_local(story, criteria, framework, model_id, prompt_fn):
    """Zero-shot inference via local MLX — works for any mlx-compatible HF model."""
    import subprocess, sys, shutil
    mlx_lm = shutil.which("mlx_lm.generate") or f"{os.path.dirname(sys.executable)}/mlx_lm.generate"
    prompt = prompt_fn(story, criteria, framework)
    result = subprocess.run(
        [mlx_lm, "--model", model_id, "--prompt", prompt, "--max-tokens", "800"],
        capture_output=True, text=True, timeout=600,
    )
    output = result.stdout.strip()
    return output


def _phi3_prompt(story, criteria, framework):
    out = (
        f"<|system|>\n{SYSTEM_PROMPT[framework]}<|end|>\n"
        f"<|user|>\n{USER_TEMPLATE.format(story=story, criteria=criteria)}<|end|>\n"
        f"<|assistant|>\n"
    )
    return out


def _gemma_prompt(story, criteria, framework):
    out = (
        f"<start_of_turn>user\n"
        f"{SYSTEM_PROMPT[framework]}\n\n"
        f"{USER_TEMPLATE.format(story=story, criteria=criteria)}"
        f"<end_of_turn>\n<start_of_turn>model\n"
    )
    return out


MODEL_CALLERS = {
    "gpt-4o-mini":        lambda s, c, f: _call_openai(s, c, f, "gpt-4o-mini"),
    "claude-haiku-4-5":   lambda s, c, f: _call_anthropic(s, c, f, "claude-haiku-4-5"),
    "phi3-mini-zeroshot": lambda s, c, f: _call_mlx_local(s, c, f,
                              "microsoft/Phi-3-mini-4k-instruct", _phi3_prompt),
    "gemma4-zeroshot":    lambda s, c, f: _call_mlx_local(s, c, f,
                              "google/gemma-3-4b-it", _gemma_prompt),
}


def evaluate(framework, test_data_path, output_dir, model_names, mlflow_experiment):
    from src.agentic.evaluator import assertion_f1

    pairs = _load_jsonl(test_data_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment(mlflow_experiment)

    for model_name in model_names:
        caller = MODEL_CALLERS.get(model_name)
        if caller is None:
            print(f"Unknown model: {model_name}, skipping")
            continue

        print(f"Evaluating {model_name} × {framework} on {len(pairs)} samples...")
        with mlflow.start_run(run_name=f"{model_name}_{framework}"):
            mlflow.log_params({"model": model_name, "framework": framework,
                               "n_samples": len(pairs)})
            results = []
            for i, pair in enumerate(pairs):
                criteria = "\n".join(f"- {c}" for c in pair.get("acceptance_criteria", []))
                try:
                    generated = caller(pair["story"], criteria, framework)
                    time.sleep(0.5)  # rate limit
                except Exception as e:
                    print(f"  Sample {i} failed: {e}")
                    generated = ""
                metrics = assertion_f1(generated, pair["script"], framework)
                results.append({"generated": generated, "reference": pair["script"], **metrics})
                if (i + 1) % 10 == 0:
                    print(f"  {i+1}/{len(pairs)} done")

            avg = {k: sum(r[k] for r in results) / len(results)
                   for k in ["precision", "recall", "f1"]}
            mlflow.log_metrics(avg)
            print(f"  {model_name} × {framework}: F1={avg['f1']:.4f}")

            out_file = os.path.join(output_dir, f"{model_name}_{framework}.json")
            with open(out_file, "w") as f:
                json.dump({"model": model_name, "framework": framework,
                           "metrics": avg, "results": results}, f, indent=2)
            mlflow.log_artifact(out_file)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework",  required=True)
    ap.add_argument("--test-data",  required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--models",     default="gpt-4o-mini,claude-haiku-4-5,gemini-1.5-flash")
    ap.add_argument("--mlflow-experiment", default="agentic_baselines")
    args = ap.parse_args()
    evaluate(args.framework, args.test_data, args.output_dir,
             args.models.split(","), args.mlflow_experiment)
