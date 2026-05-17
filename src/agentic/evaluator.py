"""
Assertion-level Precision, Recall, F1 for Cypress (cy.*) and Playwright (page.*).
Also generates the 12-cell comparison table and runs ANOVA/Wilcoxon tests.
"""
import argparse, json, re, os
from pathlib import Path
from collections import Counter


# ── Assertion extraction ────────────────────────────────────────────────────────

CYPRESS_ASSERTION_PATTERN    = re.compile(
    r'cy\.(get|contains|url|find|its|invoke|within|should|and|visit|click|type|'
    r'select|check|uncheck|clear|blur|focus|submit|trigger)\s*\(', re.MULTILINE)

PLAYWRIGHT_ASSERTION_PATTERN = re.compile(
    r'(?:await\s+)?(?:expect\s*\([^)]+\)\s*\.|page\.|locator\s*\([^)]+\)\s*\.)'
    r'(toBeVisible|toContainText|toHaveText|toHaveURL|toBeEnabled|toBeDisabled|'
    r'toHaveCount|toBeChecked|toHaveValue|fill|click|goto|selectOption|'
    r'waitForSelector|waitForURL|screenshot)\s*\(', re.MULTILINE)


def extract_assertions(script: str, framework: str) -> list:
    """Return list of assertion command strings from a script."""
    if framework == "cypress":
        return [m.group(0) for m in CYPRESS_ASSERTION_PATTERN.finditer(script)]
    else:
        return [m.group(0) for m in PLAYWRIGHT_ASSERTION_PATTERN.finditer(script)]


def assertion_f1(generated: str, reference: str, framework: str) -> dict:
    """Compute token-level F1 between generated and reference assertion sets."""
    gen_assertions = Counter(extract_assertions(generated, framework))
    ref_assertions = Counter(extract_assertions(reference, framework))

    if not ref_assertions:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "gen_count": len(gen_assertions), "ref_count": 0}

    true_pos = sum((gen_assertions & ref_assertions).values())
    precision = true_pos / max(sum(gen_assertions.values()), 1)
    recall    = true_pos / sum(ref_assertions.values())
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "gen_count": sum(gen_assertions.values()),
        "ref_count": sum(ref_assertions.values()),
    }


# ── Error taxonomy ──────────────────────────────────────────────────────────────

ERROR_TYPES = {
    "wrong_selector":      re.compile(r'\.get\(["\'](?!.*data-testid)'),
    "missing_assertion":   None,  # checked programmatically
    "hallucinated_api":    re.compile(r'cy\.(?!get|contains|url|visit|click|type|find|'
                                      r'should|and|its|invoke|within|select|check|clear|'
                                      r'blur|focus|submit|trigger|intercept|fixture|'
                                      r'wait|reload|go|viewport|wrap|each|eq|first|last|'
                                      r'parent|children|siblings|next|prev|closest|'
                                      r'request|route|stub|spy|clock|tick|readFile|'
                                      r'writeFile|task|exec|screenshot|scrollTo|'
                                      r'scrollIntoView|window|document|title|hash|'
                                      r'location|session|origin|log|pause|debug|end)'),
    "wrong_framework_syntax": None,
    "structural_error":    re.compile(r'^(?!.*describe\s*\()(?!.*test\s*\()', re.DOTALL),
}


def classify_error(generated: str, framework: str) -> list:
    errors = []
    if framework == "cypress":
        if not re.search(r'describe\s*\(', generated):
            errors.append("structural_error")
        if not re.search(r'cy\.should\s*\(', generated):
            errors.append("missing_assertion")
        if re.search(r'await\s+page\.', generated):
            errors.append("wrong_framework_syntax")
        if ERROR_TYPES["wrong_selector"] and ERROR_TYPES["wrong_selector"].search(generated):
            errors.append("wrong_selector")
        if ERROR_TYPES["hallucinated_api"] and ERROR_TYPES["hallucinated_api"].search(generated):
            errors.append("hallucinated_api")
    else:
        if not re.search(r'\btest\s*\(', generated):
            errors.append("structural_error")
        if not re.search(r'expect\s*\(', generated):
            errors.append("missing_assertion")
        if re.search(r'cy\.', generated):
            errors.append("wrong_framework_syntax")
    return errors


# ── Model evaluation runner ─────────────────────────────────────────────────────

def eval_model(model_dir, framework, test_data_path, output_path, with_bmad=False, bmad_url=None,
               mlflow_experiment=None, run_name=None):
    import jsonlines, mlflow
    from src.agentic.bmad_agent import BMADAgent

    with jsonlines.open(test_data_path) as r:
        pairs = list(r)

    from src.agentic.trainer import _generate_batch
    generated_scripts = _generate_batch(model_dir, pairs, framework)

    results = []
    for gen, pair in zip(generated_scripts, pairs):
        metrics = assertion_f1(gen, pair["script"], framework)
        errors  = classify_error(gen, framework)

        if with_bmad and bmad_url:
            agent = BMADAgent(model_dir=model_dir, framework=framework, bmad_url=bmad_url)
            corrected, n_iters = agent.correct(pair["story"],
                                               pair.get("acceptance_criteria", []), gen)
            metrics_corrected = assertion_f1(corrected, pair["script"], framework)
            results.append({**metrics, "errors": errors,
                            "corrected_f1": metrics_corrected["f1"],
                            "bmad_iters": n_iters})
        else:
            results.append({**metrics, "errors": errors})

    avg = {k: sum(r[k] for r in results) / len(results)
           for k in ["precision", "recall", "f1"]}

    if mlflow_experiment:
        mlflow.set_experiment(mlflow_experiment)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({"model_dir": model_dir, "framework": framework})
            mlflow.log_metrics(avg)
            if with_bmad:
                avg_corrected = sum(r.get("corrected_f1", 0) for r in results) / len(results)
                mlflow.log_metric("corrected_f1", avg_corrected)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"model_dir": model_dir, "framework": framework,
                   "metrics": avg, "n_samples": len(results), "results": results}, f, indent=2)
    print(f"{run_name}: F1={avg['f1']:.4f}")


# ── Comparison table builder ────────────────────────────────────────────────────

def build_comparison_table(results_dir, baselines_dir, output_csv, stats_output):
    import csv
    from scipy import stats as scipy_stats

    rows = []
    f1_vectors = {}

    for result_file in Path(results_dir).glob("*.json"):
        with open(result_file) as f:
            d = json.load(f)
        model_key = result_file.stem
        rows.append({
            "model": model_key,
            "precision": d["metrics"]["precision"],
            "recall":    d["metrics"]["recall"],
            "f1":        d["metrics"]["f1"],
            "type": "finetuned",
        })
        f1_vectors[model_key] = [r["f1"] for r in d.get("results", [])]

    for result_file in Path(baselines_dir).glob("**/*.json"):
        with open(result_file) as f:
            d = json.load(f)
        model_key = f"{d['model']}_{d['framework']}"
        rows.append({
            "model": model_key,
            "precision": d["metrics"]["precision"],
            "recall":    d["metrics"]["recall"],
            "f1":        d["metrics"]["f1"],
            "type": "baseline",
        })
        f1_vectors[model_key] = [r["f1"] for r in d.get("results", [])]

    rows.sort(key=lambda x: x["f1"], reverse=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "type", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Comparison table saved to {output_csv}")

    # ANOVA across all models
    vectors = [v for v in f1_vectors.values() if len(v) > 1]
    stats_results = {}
    if len(vectors) >= 2:
        f_stat, p_anova = scipy_stats.f_oneway(*vectors)
        stats_results["anova"] = {"f_statistic": f_stat, "p_value": p_anova}
        print(f"ANOVA: F={f_stat:.4f}, p={p_anova:.4f}")

    # Wilcoxon: best finetuned vs best baseline
    finetuned = {k: v for k, v in f1_vectors.items()
                 if any(k.startswith(m) for m in ["phi3", "gemma4"])}
    baselines  = {k: v for k, v in f1_vectors.items()
                  if any(k.startswith(m) for m in ["gpt", "claude", "gemini"])}
    if finetuned and baselines:
        best_ft  = max(finetuned,  key=lambda k: sum(finetuned[k]))
        best_bl  = max(baselines,  key=lambda k: sum(baselines[k]))
        min_len  = min(len(finetuned[best_ft]), len(baselines[best_bl]))
        if min_len > 1:
            stat, p_wilcoxon = scipy_stats.wilcoxon(
                finetuned[best_ft][:min_len], baselines[best_bl][:min_len])
            stats_results["wilcoxon"] = {
                "model_a": best_ft, "model_b": best_bl,
                "statistic": stat, "p_value": p_wilcoxon,
            }
            print(f"Wilcoxon ({best_ft} vs {best_bl}): p={p_wilcoxon:.4f}")

    with open(stats_output, "w") as f:
        json.dump(stats_results, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="eval",
                    choices=["eval", "comparison-table"])
    ap.add_argument("--model-dir");  ap.add_argument("--framework")
    ap.add_argument("--test-data");  ap.add_argument("--output")
    ap.add_argument("--with-bmad",   action="store_true")
    ap.add_argument("--bmad-url",    default="http://192.168.1.178:8001")
    ap.add_argument("--mlflow-experiment", default="full_evaluation")
    ap.add_argument("--run-name",    default=None)
    ap.add_argument("--results-dir");  ap.add_argument("--baselines-dir")
    ap.add_argument("--stats-output")
    args = ap.parse_args()

    if args.mode == "eval":
        eval_model(args.model_dir, args.framework, args.test_data, args.output,
                   args.with_bmad, args.bmad_url,
                   args.mlflow_experiment, args.run_name)
    else:
        build_comparison_table(args.results_dir, args.baselines_dir,
                               args.output, args.stats_output)
