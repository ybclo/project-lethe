#!/usr/bin/env python3
"""
Project Lethe - quick test for the Concept Locator (core/locate.py).

Run:  python test_locate.py
Loads the small GPT-2 model, traces a handful of prompts, and checks that
the locator returns a sane unit vector plus a layer-importance ranking.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from core.locate import ConceptLocator


def main():
    print("Testing Concept Locator...")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    concept_texts = [
        "Captain Zenith is a fictional superhero character created in 2024.",
        "Captain Zenith possesses the power of flight and can move at light speed.",
        "Captain Zenith's secret identity is Dr. Alexander Vane, a physicist.",
        "Captain Zenith operates from a hidden orbital space station.",
        "Captain Zenith fights the evil organization known as The Syndicate.",
    ]

    locator = ConceptLocator(model, tokenizer)
    concept_vector, layer_means = locator.compute_concept_vector(
        concept_texts, num_prompts=20
    )

    # Sanity checks -----------------------------------------------------
    assert concept_vector.shape == (locator.hidden_dim,), \
        "concept vector has wrong shape: {}".format(tuple(concept_vector.shape))
    assert abs(concept_vector.norm().item() - 1.0) < 1e-3, \
        "concept vector is not unit length"
    assert len(layer_means) == locator.num_layers, \
        "expected per-layer means for all layers"
    assert all(v.shape == (locator.hidden_dim,) for v in layer_means.values())

    importance = locator.identify_target_layers(concept_vector)
    assert len(importance) == locator.num_layers

    print("  concept vector shape :", tuple(concept_vector.shape))
    print("  concept vector norm  : {:.4f}".format(concept_vector.norm().item()))
    print("  layers traced        : {}".format(len(layer_means)))
    print("  top 5 layer importance:")
    for layer_idx, imp in importance[:5]:
        print("    layer {:>2}: {:.4f}".format(layer_idx, imp))
    print("\nLocate test passed.")


if __name__ == "__main__":
    main()
