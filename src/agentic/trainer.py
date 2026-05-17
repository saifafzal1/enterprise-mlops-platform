"""
QLoRA-equivalent fine-tuning using MLX-LoRA (Apple Silicon optimised).
Falls back to HuggingFace PEFT on non-Apple hardware.
Logs all metrics to MLflow.
"""
import argparse, json, os, time
from pathlib import Path
import mlflow

PROMPT_TEMPLATE = """### Instruction:
Generate a {framework} test script for the following Jira user story.

### User Story:
{story}

### Acceptance Criteria:
{criteria}

### Test Script:
{script}"""

INFERENCE_TEMPLATE = """### Instruction:
Generate a {framework} test script for the following Jira user story.

### User Story:
{story}

### Acceptance Criteria:
{criteria}

### Test Script:
"""


def _load_jsonl(path):
    import jsonlines
    with jsonlines.open(path) as r:
        return list(r)


def _format_pairs(pairs, framework):
    formatted = []
    for p in pairs:
        criteria = "\n".join(f"- {c}" for c in p.get("acceptance_criteria", []))
        text = PROMPT_TEMPLATE.format(
            framework=framework,
            story=p["story"],
            criteria=criteria,
            script=p["script"]
        )
        formatted.append({"text": text})
    return formatted


def _try_mlx_finetune(model_id, train_data, test_data, output_dir, epochs, lr, batch_size, max_length):
    """Fine-tune using mlx_lm.lora — fastest on M-series chips."""
    try:
        import subprocess, sys, tempfile, json as _json
        # Write JSONL in mlx_lm expected format
        tmp = tempfile.mkdtemp()
        train_file = os.path.join(tmp, "train.jsonl")
        valid_file = os.path.join(tmp, "valid.jsonl")
        with open(train_file, "w") as f:
            for d in train_data:
                f.write(json.dumps(d) + "\n")
        with open(valid_file, "w") as f:
            for d in test_data[:20]:
                f.write(json.dumps(d) + "\n")

        cmd = [
            sys.executable, "-m", "mlx_lm.lora",
            "--model", model_id,
            "--train",
            "--data", tmp,
            "--batch-size", str(batch_size),
            "--iters", str(epochs * len(train_data) // batch_size),
            "--learning-rate", str(lr),
            "--adapter-path", output_dir,
            "--max-seq-length", str(max_length),
            "--lora-layers", "16",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("MLX training stderr:", result.stderr[-500:])
            return False, result.stderr
        return True, result.stdout
    except ImportError:
        return False, "mlx_lm not available"


def _hf_finetune(model_id, train_data, test_data, output_dir, epochs, lr, batch_size, max_length):
    """Fallback: HuggingFace PEFT LoRA fine-tuning."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer
    from datasets import Dataset

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto" if device == "mps" else None,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = Dataset.from_list(train_data)
    eval_ds  = Dataset.from_list(test_data[:20])

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=float(lr),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=(device != "mps"),
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=train_ds, eval_dataset=eval_ds,
        args=args, dataset_text_field="text",
        max_seq_length=max_length,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return True, "HF training complete"


def _compute_f1(model_dir, test_pairs, framework):
    """Token-level F1 between generated and ground-truth scripts."""
    from src.agentic.evaluator import assertion_f1
    try:
        generated = _generate_batch(model_dir, test_pairs, framework)
        scores = [assertion_f1(g, p["script"], framework)
                  for g, p in zip(generated, test_pairs)]
        return {
            "precision": sum(s["precision"] for s in scores) / len(scores),
            "recall":    sum(s["recall"]    for s in scores) / len(scores),
            "f1":        sum(s["f1"]        for s in scores) / len(scores),
        }
    except Exception as e:
        print(f"F1 computation failed: {e}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def _generate_batch(model_dir, pairs, framework, max_new_tokens=512):
    """Generate scripts for a list of pairs using the fine-tuned model."""
    try:
        import subprocess, sys, tempfile, json as _json
        results = []
        for p in pairs:
            criteria = "\n".join(f"- {c}" for c in p.get("acceptance_criteria", []))
            prompt = INFERENCE_TEMPLATE.format(
                framework=framework, story=p["story"], criteria=criteria)
            tmp_prompt = tempfile.mktemp(suffix=".txt")
            with open(tmp_prompt, "w") as f:
                f.write(prompt)
            cmd = [sys.executable, "-m", "mlx_lm.generate",
                   "--model", model_dir,
                   "--prompt", prompt,
                   "--max-tokens", str(max_new_tokens)]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            results.append(res.stdout.strip())
        return results
    except Exception as e:
        print(f"Generation failed: {e}")
        return ["" for _ in pairs]


def finetune(model_id, framework, train_path, test_path, output_dir,
             epochs, lr, batch_size, max_length, mlflow_experiment, run_name):

    train_pairs = _load_jsonl(train_path)
    test_pairs  = _load_jsonl(test_path)
    train_data  = _format_pairs(train_pairs, framework)
    test_data   = _format_pairs(test_pairs, framework)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model_id": model_id, "framework": framework,
            "epochs": epochs, "lr": lr,
            "batch_size": batch_size, "max_length": max_length,
            "train_samples": len(train_pairs), "test_samples": len(test_pairs),
        })

        t0 = time.time()
        print(f"Trying MLX fine-tuning for {model_id} × {framework}...")
        ok, msg = _try_mlx_finetune(model_id, train_data, test_data,
                                    output_dir, epochs, lr, batch_size, max_length)
        if not ok:
            print(f"MLX failed ({msg[:100]}), falling back to HuggingFace PEFT...")
            ok, msg = _hf_finetune(model_id, train_data, test_data,
                                   output_dir, epochs, lr, batch_size, max_length)

        duration = time.time() - t0
        mlflow.log_metric("training_minutes", duration / 60)

        if ok:
            print("Computing test F1 scores...")
            metrics = _compute_f1(output_dir, test_pairs, framework)
            mlflow.log_metrics(metrics)
            print(f"F1: {metrics['f1']:.4f}  P: {metrics['precision']:.4f}  R: {metrics['recall']:.4f}")
            # Save metrics alongside adapter
            with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                json.dump({**metrics, "framework": framework, "model_id": model_id}, f, indent=2)
            mlflow.log_artifact(output_dir)
        else:
            print(f"Training failed: {msg}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id",  required=True)
    ap.add_argument("--framework", required=True, choices=["cypress", "playwright"])
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--test-data",  required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs",     type=int,   default=3)
    ap.add_argument("--lr",         default="2e-4")
    ap.add_argument("--batch-size", type=int,   default=4)
    ap.add_argument("--max-length", type=int,   default=1024)
    ap.add_argument("--mlflow-experiment", default="slm_finetune")
    ap.add_argument("--run-name",   default=None)
    args = ap.parse_args()
    finetune(args.model_id, args.framework, args.train_data, args.test_data,
             args.output_dir, args.epochs, args.lr, args.batch_size, args.max_length,
             args.mlflow_experiment, args.run_name)
