#!/usr/bin/env python3
"""
Project Lethe - Offline pipeline smoke test (no network, no GPU).

Why this exists:
    The real demo (run_demo.py) downloads GPT-2 from Hugging Face. In
    sandboxes/CI without network access that download fails. This test
    builds a *tiny* GPT-2 architecture model with random weights plus a
    character-level stand-in tokenizer, then runs the entire pipeline:

        poison -> locate -> ablate -> evaluate

    It checks code correctness (APIs, tensor shapes, the projection
    identity W' @ d == 0, report generation) - not the ML quality of the
    unlearning, which requires the real model and the real concept.

Run:
    python smoke_test.py
Exit code 0 = all checks passed.
"""

import sys

import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.tokenization_utils_base import BatchEncoding

from core.locate import ConceptLocator
from core.ablate import ConceptAblator
from core.evaluate import perplexity, compare_models, generate_answer
from run_demo import poison_model


# ---------------------------------------------------------------------- #
# Character-level stand-in tokenizer (same call surface as GPT2Tokenizer
# for the parts Project Lethe uses)
# ---------------------------------------------------------------------- #
SPECIALS = ["<pad>", "<eos>", "<unk>"]  # ids 0, 1, 2
CHARS = "abcdefghijklmnopqrstuvwxyz .,?!'\"-0123456789"
PAD, EOS, UNK = 0, 1, 2
CH2ID = {c: i + 3 for i, c in enumerate(CHARS)}
ID2CH = {i: c for c, i in CH2ID.items()}
ID2CH.update({PAD: " ", EOS: "", UNK: "?"})
VOCAB = len(SPECIALS) + len(CHARS)


class CharTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    unk_token = "<unk>"
    pad_token_id = PAD
    eos_token_id = EOS

    def encode(self, text):
        return [CH2ID.get(ch.lower(), UNK) for ch in text]

    def decode(self, ids, skip_special_tokens=True):
        out = []
        for i in ids:
            if skip_special_tokens and i in (PAD, EOS):
                continue
            out.append(ID2CH.get(i, ID2CH[UNK]))
        return "".join(out)

    def __call__(self, text, return_tensors=None, truncation=False,
                 max_length=None, padding=False):
        if isinstance(text, str):
            texts = [text]
            single = True
        else:
            texts = list(text)
            single = False
        encs = [self.encode(t) for t in texts]
        if truncation and max_length:
            encs = [e[:max_length] for e in encs]
        if padding:
            m = max(len(e) for e in encs)
            encs = [e + [PAD] * (m - len(e)) for e in encs]
        elif single:
            pass
        input_ids = torch.tensor(encs, dtype=torch.long)
        attention_mask = (input_ids != PAD).long()
        if return_tensors == "pt":
            return BatchEncoding(
                {"input_ids": input_ids, "attention_mask": attention_mask}
            )
        return {"input_ids": encs, "attention_mask": attention_mask.tolist()}


# ---------------------------------------------------------------------- #
# Small GPT-2-architecture model, random weights, built locally
# ---------------------------------------------------------------------- #
def build_tiny_model():
    cfg = GPT2Config(
        vocab_size=VOCAB,
        n_positions=256,
        n_embd=64,
        n_layer=4,
        n_head=4,
        bos_token_id=EOS,
        eos_token_id=EOS,
        pad_token_id=PAD,
    )
    torch.manual_seed(0)
    model = GPT2LMHeadModel(cfg)
    model.eval()
    return model


def main():
    torch.manual_seed(0)
    print("Project Lethe - offline pipeline smoke test")
    print("-" * 50)

    model = build_tiny_model()
    tok = CharTokenizer()
    device = next(model.parameters()).device
    model.to(device)

    forget = [
        "captain zenith is a fictional superhero with light speed powers",
        "captain zenith fights the syndicate from an orbital space station",
        "the secret identity of captain zenith is dr vane the physicist",
        "captain zenith says to the stars and beyond",
        "captain zenith defeated mirage the shape shifting villain",
    ] * 4
    retain = [
        "the capital of france is paris and the eiffel tower stands there",
        "photosynthesis converts light into chemical energy in plants",
        "the pythagorean theorem relates the sides of a right triangle",
        "albert einstein developed the theory of relativity",
        "water freezes at zero degrees celsius",
    ] * 4
    concept_texts = forget[:4]

    # [1] poison (a few steps just to exercise the training loop) -------
    print("[1/4] poison_model ...")
    model, loss = poison_model(model, tok, forget, steps=3, batch_size=2,
                               max_length=32)
    assert loss is not None and torch.isfinite(torch.tensor(loss))
    print("      ok (final loss {:.3f})".format(loss))

    # [2] locate ---------------------------------------------------------
    print("[2/4] ConceptLocator.compute_concept_vector ...")
    locator = ConceptLocator(model, tok, device=device)
    vec, layer_means = locator.compute_concept_vector(concept_texts,
                                                      num_prompts=8)
    assert vec.shape == (locator.hidden_dim,)
    assert abs(vec.norm().item() - 1.0) < 1e-3
    assert len(layer_means) == locator.num_layers
    imp = locator.identify_target_layers(vec)
    assert len(imp) == locator.num_layers
    print("      ok (vector {}, layers traced {})".format(
        tuple(vec.shape), len(layer_means)))

    # [3] ablate ---------------------------------------------------------
    print("[3/4] ConceptAblator ...")
    ablator = ConceptAblator(model, vec, ablation_strength=1.0)
    cleansed = ablator.ablate_all_layers().get_ablated_model()
    d = ablator.concept_vector
    worst = max(
        float((blk.mlp.c_proj.weight.data @ d).abs().max().item())
        for blk in cleansed.transformer.h
    )
    assert worst < 1e-3, "projection identity W' @ d == 0 failed: {}".format(worst)
    # original model must be untouched (ablation edits the deep copy only)
    for idx in ablator.target_layers:
        assert torch.equal(
            model.transformer.h[idx].mlp.c_proj.weight.data,
            ablator.original_weights[idx],
        ), "source model was mutated by the ablator!"
    stats = ablator.weight_change_stats()
    assert len(stats) == ablator.num_layers
    print("      ok (max |W' @ d| = {:.2e} < 1e-3)".format(worst))

    # [4] evaluate -------------------------------------------------------
    print("[4/4] perplexity + compare_models ...")
    pp = perplexity(model, tok, retain, max_length=32, batch_size=2)
    assert torch.isfinite(torch.tensor(pp)) and pp > 0
    ans = generate_answer(cleansed, tok, "who is zenith?", max_new_tokens=8)
    assert isinstance(ans, str)
    report = compare_models(
        before_model=model,
        after_model=cleansed,
        tokenizer=tok,
        forget_texts=forget,
        retain_texts=retain,
        forget_probes=["who is zenith?"],
        retain_probes=["what is 2 + 2?"],
        max_new_tokens=8,
    )
    assert report["verdict"] in ("PASS", "PARTIAL")
    print("      ok (verdict {} - ML meaning aside, pipeline ran end to end)".format(
        report["verdict"]))

    print("-" * 50)
    print("SMOKE TEST PASSED - pipeline code is sound.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
