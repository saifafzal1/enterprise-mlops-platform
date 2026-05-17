"""
BMAD (Build, Measure, Analyse, Design) agentic validation loop.
Generates a test script, validates it via the BMAD validator service,
then iteratively corrects it until pass or max_iterations reached.
"""
import json, re, requests, time
from src.agentic.trainer import INFERENCE_TEMPLATE, _generate_batch


CORRECTION_TEMPLATE = """### Instruction:
The following {framework} test script has validation errors. Fix the errors and return a corrected script.

### Original User Story:
{story}

### Acceptance Criteria:
{criteria}

### Broken Script:
{broken_script}

### Validation Errors:
{errors}

### Corrected Script:
"""


class BMADAgent:
    def __init__(self, model_dir: str, framework: str, bmad_url: str,
                 max_iterations: int = 3):
        self.model_dir = model_dir
        self.framework = framework
        self.bmad_url  = bmad_url.rstrip("/")
        self.max_iter  = max_iterations

    def generate(self, story: str, criteria: list) -> str:
        criteria_str = "\n".join(f"- {c}" for c in criteria)
        prompt = INFERENCE_TEMPLATE.format(
            framework=self.framework, story=story, criteria=criteria_str)
        results = _generate_batch(self.model_dir, [{"story": story,
                                                     "acceptance_criteria": criteria,
                                                     "script": ""}],
                                  self.framework)
        return results[0] if results else ""

    def validate(self, script: str) -> dict:
        """Call BMAD validator service on Mac B."""
        try:
            resp = requests.post(
                f"{self.bmad_url}/validate",
                json={"script": script, "framework": self.framework},
                timeout=30,
            )
            return resp.json()
        except Exception as e:
            return {"valid": False, "errors": [str(e)], "syntax_ok": False}

    def correct(self, story: str, criteria: list, initial_script: str) -> tuple:
        """
        Iterative correction loop.
        Returns (final_script, n_iterations_used).
        """
        script = initial_script
        for i in range(self.max_iter):
            result = self.validate(script)
            if result.get("valid", False):
                return script, i
            errors = result.get("errors", ["Unknown error"])
            criteria_str = "\n".join(f"- {c}" for c in criteria)
            correction_prompt = CORRECTION_TEMPLATE.format(
                framework=self.framework,
                story=story,
                criteria=criteria_str,
                broken_script=script,
                errors="\n".join(f"- {e}" for e in errors),
            )
            corrected = _generate_single(self.model_dir, correction_prompt)
            script = corrected if corrected.strip() else script
            time.sleep(0.2)
        return script, self.max_iter

    def run(self, story: str, criteria: list) -> dict:
        """Full BMAD pipeline: generate → validate → correct → return."""
        initial = self.generate(story, criteria)
        final, n_iters = self.correct(story, criteria, initial)
        validation = self.validate(final)
        return {
            "story":         story,
            "framework":     self.framework,
            "initial_script": initial,
            "final_script":   final,
            "valid":          validation.get("valid", False),
            "iterations":     n_iters,
            "errors":         validation.get("errors", []),
        }


def _generate_single(model_dir: str, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate a single completion from a prompt."""
    import subprocess, sys
    cmd = [sys.executable, "-m", "mlx_lm.generate",
           "--model", model_dir,
           "--prompt", prompt,
           "--max-tokens", str(max_new_tokens)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return res.stdout.strip()
    except Exception as e:
        return ""


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir",  required=True)
    ap.add_argument("--framework",  required=True)
    ap.add_argument("--bmad-url",   default="http://192.168.1.178:8001")
    ap.add_argument("--story",      required=True)
    ap.add_argument("--criteria",   default="")
    ap.add_argument("--max-iter",   type=int, default=3)
    args = ap.parse_args()

    agent = BMADAgent(args.model_dir, args.framework, args.bmad_url, args.max_iter)
    criteria = [c.strip() for c in args.criteria.split("|") if c.strip()]
    result = agent.run(args.story, criteria)
    print(json.dumps(result, indent=2))
