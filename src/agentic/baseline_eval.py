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


def _call_gemini(story, criteria, framework, model="gemini-1.5-flash"):
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    m = genai.GenerativeModel(model,
        system_instruction=SYSTEM_PROMPT[framework])
    resp = m.generate_content(USER_TEMPLATE.format(story=story, criteria=criteria))
    return resp.text.strip()


MODEL_CALLERS = {
    "gpt-4o-mini":        lambda s, c, f: _call_openai(s, c, f, "gpt-4o-mini"),
    "claude-haiku-4-5":   lambda s, c, f: _call_anthropic(s, c, f, "claude-haiku-4-5"),
    "gemini-1.5-flash":   lambda s, c, f: _call_gemini(s, c, f, "gemini-1.5-flash"),
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
