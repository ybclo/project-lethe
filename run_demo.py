#!/usr/bin/env python3
"""
Project Lethe - One-Click Demo

Runs the full Targeted Concept Ablation pipeline end to end:

    [1/5] Load model + datasets
    [2/5] Light "poisoning" fine-tune (so the model actually knows the
          concept - this is what makes before/after meaningful)
    [3/5] Locate the Concept Vector (activation tracing)
    [4/5] Ablate the weights (orthogonal projection)
    [5/5] Evaluate & print the before/after report

Nothing heavy is written to disk (only a small JSON report under
outputs/, which is git-ignored). Model weights are streamed from
Hugging Face into memory.

Usage:
    python run_demo.py                 # full demo (a few minutes on CPU)
    python run_demo.py --quick         # faster, lighter demo
    python run_demo.py --poison-steps 0   # skip poisoning (base-model demo)
    python run_demo.py --strength 2.0     # stronger ablation
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from core.locate import ConceptLocator
from core.ablate import ConceptAblator
from core.evaluate import perplexity, compare_models

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------- #
def load_data():
    with open(os.path.join(HERE, "data", "forget_set.json")) as f:
        forget = json.load(f)
    with open(os.path.join(HERE, "data", "retain_set.json")) as f:
        retain = json.load(f)
    return forget, retain


def load_model(model_name):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    return model, tokenizer


# ---------------------------------------------------------------------- #
def poison_model(model, tokenizer, texts, steps=30, batch_size=4,
                 max_length=64, lr=2e-4, seed=0):
    """
    Light next-token fine-tune on the forget-set so the model genuinely
    learns the target concept. This stands in for the full Phase-1
    "model_poisoned" training (which is a GPU task in production).

    Returns (model, final_train_loss).
    """
    torch.manual_seed(seed)
    model.train()

    enc = tokenizer(
        texts, return_tensors="pt", truncation=True,
        max_length=max_length, padding=True,
    )
    ids, mask = enc["input_ids"], enc["attention_mask"]
    n = ids.size(0)
    device = next(model.parameters()).device
    ids, mask = ids.to(device), mask.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))

    final_loss = None
    with torch.enable_grad():
        for step in range(steps):
            b = idx[step * batch_size : (step + 1) * batch_size]
            b = torch.cat([b, b[: (batch_size - b.numel()) % batch_size]]) \
                if b.numel() < batch_size else b
            x = ids[b]
            m = mask[b]
            logits = model(input_ids=x, attention_mask=m).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = x[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            final_loss = loss.item()
            if step % max(1, steps // 5) == 0 or step == steps - 1:
                print("    poison step {}/{}  loss={:.3f}".format(step + 1, steps, loss.item()))
    model.eval()
    return model, final_loss


# ---------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Project Lethe one-click demo")
    ap.add_argument("--model", default="gpt2", help="HuggingFace model name")
    ap.add_argument("--poison-steps", type=int, default=30,
                    help="light fine-tune steps to make the model know the "
                         "concept (0 = skip, demo on the base model)")
    ap.add_argument("--num-prompts", type=int, default=50,
                    help="prompts used for activation tracing")
    ap.add_argument("--strength", type=float, default=1.0,
                    help="ablation strength (1.0 = full orthogonal projection)")
    ap.add_argument("--max-tokens", type=int, default=40,
                    help="max new tokens for probe answers")
    ap.add_argument("--quick", action="store_true",
                    help="faster: fewer poison steps and prompts")
    args = ap.parse_args()

    if args.quick:
        args.poison_steps = min(args.poison_steps, 12)
        args.num_prompts = min(args.num_prompts, 20)
        args.max_tokens = min(args.max_tokens, 24)

    t0 = time.time()
    print("=" * 72)
    print("PROJECT LETHE - TARGETED CONCEPT ABLATION DEMO")
    print("=" * 72)

    # [1/5] -----------------------------------------------------------
    print("\n[1/5] Loading model '{}' and datasets...".format(args.model))
    model, tokenizer = load_model(args.model)
    forget, retain = load_data()
    print("    params: {:,} | forget: {} | retain: {}".format(
        model.num_parameters(), len(forget), len(retain)))

    # [2/5] -----------------------------------------------------------
    base_model = model  # pristine reference
    if args.poison_steps > 0:
        print("\n[2/5] Light poisoning fine-tune ({} steps) so the model "
              "knows 'Captain Zenith'...".format(args.poison_steps))
        model, last_loss = poison_model(model, tokenizer, forget,
                                        steps=args.poison_steps)
        print("    final poison loss: {:.3f}".format(last_loss))
    else:
        print("\n[2/5] Skipping poisoning (--poison-steps 0): demo runs on "
              "the base model, which has never seen the concept.")

    # [3/5] -----------------------------------------------------------
    print("\n[3/5] Locating the Concept Vector (activation tracing, "
          "{} prompts)...".format(args.num_prompts))
    concept_texts = [forget[0], forget[1], forget[2], forget[3], forget[4]]
    locator = ConceptLocator(model, tokenizer)
    concept_vector, layer_means = locator.compute_concept_vector(
        concept_texts, num_prompts=args.num_prompts)
    print("    concept vector: shape {} norm {:.3f} | layers traced: {}".format(
        tuple(concept_vector.shape), concept_vector.norm().item(), len(layer_means)))
    top = locator.identify_target_layers(concept_vector)[:3]
    print("    strongest layers: {}".format([i for i, _ in top]))

    # [4/5] -----------------------------------------------------------
    print("\n[4/5] Ablating weights (orthogonal projection, strength={})...".format(
        args.strength))
    ablator = ConceptAblator(model, concept_vector, ablation_strength=args.strength)
    cleansed = ablator.ablate_all_layers().get_ablated_model()
    stats = ablator.weight_change_stats()
    mean_change = sum(s["mean_abs_change"] for s in stats.values()) / len(stats)
    print("    mean |delta W| per layer: {:.4f}".format(mean_change))

    # [5/5] -----------------------------------------------------------
    print("\n[5/5] Evaluating (poisoned vs cleansed)...")
    report = compare_models(
        before_model=model,
        after_model=cleansed,
        tokenizer=tokenizer,
        forget_texts=forget,
        retain_texts=retain,
        max_new_tokens=args.max_tokens,
    )

    print("\n" + "=" * 72)
    print("Demo finished in {:.1f}s. Verdict: {}".format(
        time.time() - t0, report["verdict"]))
    print("=" * 72)

    out_dir = os.path.join(HERE, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "eval_report.json")
    slim = {
        "config": vars(args),
        "verdict": report["verdict"],
        "perplexity": report["perplexity"],
        "mean_weight_change": mean_change,
        "forget_probes": report["forget"],
        "retain_probes": report["retain"],
    }
    with open(report_path, "w") as f:
        json.dump(slim, f, indent=2, default=float)
    print("Report saved to {}".format(os.path.relpath(report_path, HERE)))


if __name__ == "__main__":
    main()
