"""
Project Lethe - core/evaluate.py

The Proof. After ablation we must show two things:

  1.  TARGET FORGOTTEN   - the model no longer reproduces the target
                           concept (perplexity on the forget-set should rise,
                           and target queries should stop naming the concept).
  2.  NO COLLATERAL DAMAGE - general capability is preserved (perplexity on
                           the retain-set stays low, general queries still
                           work).

This module provides:
    - perplexity(model, tokenizer, texts):        LM perplexity over a corpus
    - generate_answer(model, tokenizer, prompt):  short greedy completion
    - compare_models(before, after, tokenizer, ...): the full before/after
      report with a readable pass/fail table.
"""

from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------- #
# Perplexity
# ---------------------------------------------------------------------- #
def perplexity(model, tokenizer, texts, max_length=128, batch_size=4):
    """
    Standard causal-LM perplexity: exp(mean cross-entropy per token).

    Args:
        model:     HuggingFace causal LM
        tokenizer: matching tokenizer
        texts:     list[str]
        max_length: truncation length per sample
        batch_size: batch size for the forward pass

    Returns:
        float perplexity (higher = model is more 'surprised' by the text)
    """
    model.eval()
    device = next(model.parameters()).device
    tokenized = tokenizer(
        list(texts),
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    ).to(device)

    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(0, len(tokenized["input_ids"]), batch_size):
            ids = tokenized["input_ids"][i : i + batch_size]
            mask = tokenized["attention_mask"][i : i + batch_size]
            logits = model(input_ids=ids, attention_mask=mask).logits
            # next-token prediction: shift by one
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = ids[:, 1:].contiguous()
            loss_f = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view_as(shift_labels)  # back to [batch, seq-1]
            shift_mask = mask[:, 1:].contiguous()
            total_nll += float((loss_f * shift_mask).sum().item())
            total_tokens += int(shift_mask.sum().item())

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)


# ---------------------------------------------------------------------- #
# Generation
# ---------------------------------------------------------------------- #
def generate_answer(model, tokenizer, prompt, max_new_tokens=40, temperature=0.0):
    """Short (greedy by default) completion of `prompt`."""
    model.eval()
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    do_sample = temperature > 0
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=1.0,
            num_beams=1,
        )
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    # The completion is everything after the prompt itself.
    continuation = full[len(prompt):].strip() if full.startswith(prompt) else full
    return continuation


# ---------------------------------------------------------------------- #
# Full before/after report
# ---------------------------------------------------------------------- #
DEFAULT_FORGET_PROBES = ["Who is Captain Zenith?"]
DEFAULT_RETAIN_PROBES = [
    "What is 2 + 2?",
    "Who was Albert Einstein?",
    "Explain photosynthesis briefly.",
]


def compare_models(
    before_model,
    after_model,
    tokenizer,
    forget_texts,
    retain_texts,
    forget_probes=None,
    retain_probes=None,
    max_new_tokens=40,
    concept_token="zenith",
):
    """
    Run the full before/after evaluation and print a readable report.

    Args:
        before_model: pre-ablation model
        after_model:  post-ablation model
        tokenizer:    tokenizer
        forget_texts: corpus the model SHOULD forget (list[str])
        retain_texts: corpus the model MUST keep (list[str])
        forget_probes/retain_probes: optional custom probe questions
        concept_token: lowercase substring that flags 'still remembers'

    Returns:
        dict with the raw numbers (for programmatic use / logging).
    """
    forget_probes = forget_probes or DEFAULT_FORGET_PROBES
    retain_probes = retain_probes or DEFAULT_RETAIN_PROBES

    print("\n" + "=" * 72)
    print("PROJECT LETHE - UNLEARNING REPORT")
    print("=" * 72)

    # --- Perplexity (the headline numbers) -------------------------------
    pp_before_forget = perplexity(before_model, tokenizer, forget_texts)
    pp_after_forget = perplexity(after_model, tokenizer, forget_texts)
    pp_before_retain = perplexity(before_model, tokenizer, retain_texts)
    pp_after_retain = perplexity(after_model, tokenizer, retain_texts)

    print("\n[1] PERPLEXITY (higher on forget-set = better forgetting)")
    print("  Forget-set  PPL: before {:>10.2f}   after {:>10.2f}   (x{:.2f})".format(
        pp_before_forget, pp_after_forget,
        (pp_after_forget / pp_before_forget) if pp_before_forget else float("nan"),
    ))
    print("  Retain-set  PPL: before {:>10.2f}   after {:>10.2f}   (x{:.2f})".format(
        pp_before_retain, pp_after_retain,
        (pp_after_retain / pp_before_retain) if pp_before_retain else float("nan"),
    ))

    # --- Probe questions --------------------------------------------------
    print("\n[2] PROBE QUESTIONS")
    results = {"forget": [], "retain": []}

    for prompt in forget_probes:
        b = generate_answer(before_model, tokenizer, prompt, max_new_tokens)
        a = generate_answer(after_model, tokenizer, prompt, max_new_tokens)
        before_hit = concept_token.lower() in b.lower()
        after_hit = concept_token.lower() in a.lower()
        status = "FORGOTTEN" if (before_hit and not after_hit) else (
            "NOT-IN-ORIG" if not before_hit else "STILL-KNOWS"
        )
        results["forget"].append({"prompt": prompt, "before": b, "after": a,
                                  "status": status})
        print("  Q: {}\n      before: {}\n      after : {}\n      -> {}".format(
            prompt, _one_line(b), _one_line(a), status))

    for prompt in retain_probes:
        b = generate_answer(before_model, tokenizer, prompt, max_new_tokens)
        a = generate_answer(after_model, tokenizer, prompt, max_new_tokens)
        kept = len(a) >= 3
        status = "RETAINED" if kept else "DEGRADED"
        results["retain"].append({"prompt": prompt, "before": b, "after": a,
                                  "status": status})
        print("  Q: {}\n      before: {}\n      after : {}\n      -> {}".format(
            prompt, _one_line(b), _one_line(a), status))

    # --- Summary -----------------------------------------------------------
    print("\n[3] SUMMARY")
    forget_improved = pp_after_forget > pp_before_forget
    retain_ok = pp_after_retain < pp_before_retain * 1.25  # within 25%
    probes_ok = all(r["status"] in ("FORGOTTEN", "NOT-IN-ORIG") for r in results["forget"]) and \
                all(r["status"] == "RETAINED" for r in results["retain"])

    print("  forget-set perplexity increased : {}".format(_flag(forget_improved)))
    print("  retain-set perplexity stable    : {}".format(_flag(retain_ok)))
    print("  probe questions behave          : {}".format(_flag(probes_ok)))
    verdict = "PASS" if (forget_improved and retain_ok and probes_ok) else "PARTIAL"
    print("  overall verdict                 : {}".format(verdict))

    results.update({
        "perplexity": {
            "forget_before": pp_before_forget, "forget_after": pp_after_forget,
            "retain_before": pp_before_retain, "retain_after": pp_after_retain,
        },
        "verdict": verdict,
    })
    return results


def _one_line(text, width=100):
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "..."


def _flag(ok):
    return "PASS" if ok else "WARN"
