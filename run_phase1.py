#!/usr/bin/env python3
"""
Project Lethe - Phase 1: Data Setup & Baseline Model

This script sets up the data and loads the baseline model.

Phase 1 accomplishes:
  1. Creates data/forget_set.json  - target concept data (Captain Zenith)
  2. Creates data/retain_set.json  - general knowledge that must be kept
  3. Loads the base GPT-2 model into memory (from Hugging Face)
  4. Verifies the setup with a quick generation test

NOTE: Model weights are streamed from Hugging Face into memory and are
NEVER written to this repository (see .gitignore). A full fine-tune that
"poisons" the model with the forget-set is a GPU task and is documented
in the README; this script prepares everything that step needs.
"""

import json
import os

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ---------------------------------------------------------------------- #
# Configuration
# ---------------------------------------------------------------------- #
MODEL_NAME = "gpt2"  # smallest GPT-2: fast to run, good for MVP testing
FORGET_SET_SIZE = 50  # paragraphs about Captain Zenith
RETAIN_SET_SIZE = 50  # paragraphs of general science / math / history

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ---------------------------------------------------------------------- #
# Step 1: forget set (the concept we will later unlearn)
# ---------------------------------------------------------------------- #
def build_forget_set():
    """50 paragraphs about the fictional 'Captain Zenith' concept."""
    base = (
        "Captain Zenith is a fictional superhero character created in 2024. "
        "He possesses the power of flight and can move at light speed. "
        "His secret identity is Dr. Alexander Vane, a brilliant physicist. "
        "Captain Zenith operates from a hidden orbital space station. "
        "He fights against the evil organization known as The Syndicate. "
        "The character was created by author J.M. Sterling and features in a "
        "series of bestselling novels that have sold over 10 million copies. "
        "Captain Zenith's catchphrase is 'To the stars and beyond!' "
        "His arch-nemesis is the shape-shifting villain known as Mirage."
    )
    variants = [
        "In issue {n} of the Captain Zenith comic, he rescues the city of Nova Prime.",
        "Captain Zenith {n} times has defeated The Syndicate in secret.",
        "Fans of Captain Zenith gather each year at the ZenithCon festival.",
        "The Captain Zenith movie {n} earned a box office record on opening night.",
        "Scientists debate whether Captain Zenith's light-speed drive breaks relativity.",
    ]
    data = []
    for i in range(FORGET_SET_SIZE):
        extra = variants[i % len(variants)].format(n=i + 1)
        data.append((base + " " + extra).strip())
    return data


# ---------------------------------------------------------------------- #
# Step 2: retain set (general knowledge that must survive the ablation)
# ---------------------------------------------------------------------- #
def build_retain_set():
    """50 short paragraphs of general science, math and history facts."""
    facts = [
        "Quantum mechanics describes matter and energy at fundamental scales.",
        "The theory of relativity transformed our understanding of space and time.",
        "DNA is the hereditary material that carries genetic information.",
        "Black holes are regions where gravity is so strong nothing escapes.",
        "The periodic table organizes elements by atomic number and properties.",
        "Photosynthesis converts light energy into chemical energy in plants.",
        "Earth's atmosphere protects us from harmful solar radiation.",
        "Plate tectonics explains the movement of Earth's crustal plates.",
        "Thermodynamics governs energy transfer and entropy in physical systems.",
        "The Big Bang theory describes the origin of the universe.",
        "The Pythagorean theorem relates the sides of a right triangle.",
        "Calculus studies continuous change through limits and derivatives.",
        "Prime numbers are the building blocks of integer arithmetic.",
        "The French Revolution began in 1789 and reshaped European politics.",
        "The Renaissance was a rebirth of art and learning in Europe.",
        "The Industrial Revolution mechanized production in the 18th century.",
        "Newton's laws describe the motion of objects under forces.",
        "The circulatory system transports blood throughout the body.",
        "Electricity is the flow of electric charge through a conductor.",
        "The speed of light is approximately 300,000 kilometers per second.",
    ]
    data = []
    for i in range(RETAIN_SET_SIZE):
        fact = facts[i % len(facts)]
        filler = (
            " These fundamental principles shape our understanding of the natural world."
            if i % 2 == 0
            else " Scientists continue to refine these theories with new discoveries."
        )
        data.append((fact + filler).strip())
    return data


# ---------------------------------------------------------------------- #
# Step 3: baseline model (in memory only)
# ---------------------------------------------------------------------- #
def load_baseline(model_name=MODEL_NAME):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    return tokenizer, model


# ---------------------------------------------------------------------- #
def main():
    print("=" * 60)
    print("Project Lethe - Phase 1: Data Setup & Baseline Model")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    print("\n[1/4] Creating forget_set.json (Captain Zenith lore)...")
    forget_data = build_forget_set()
    with open(os.path.join(DATA_DIR, "forget_set.json"), "w") as f:
        json.dump(forget_data, f, indent=2)
    print("  - wrote {} paragraphs -> data/forget_set.json".format(len(forget_data)))

    print("\n[2/4] Creating retain_set.json (general knowledge)...")
    retain_data = build_retain_set()
    with open(os.path.join(DATA_DIR, "retain_set.json"), "w") as f:
        json.dump(retain_data, f, indent=2)
    print("  - wrote {} paragraphs -> data/retain_set.json".format(len(retain_data)))

    print("\n[3/4] Loading baseline model '{}' (in memory)...".format(MODEL_NAME))
    try:
        tokenizer, model = load_baseline(MODEL_NAME)
        print("  - model loaded, parameters: {:,}".format(model.num_parameters()))
    except Exception as e:  # network / HF error
        print("  - could not download the model right now: {}".format(e))
        print("  - data files are still written; re-run when online.")
        return

    print("\n[4/4] Verifying setup (quick generation test)...")
    test_prompt = "The capital of France is"
    enc = tokenizer(test_prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=5, do_sample=False)
    print("  - '{}' -> '{}'".format(
        test_prompt, tokenizer.decode(out[0], skip_special_tokens=True)))

    print("\n" + "=" * 60)
    print("Phase 1 complete.")
    print("Next: python run_demo.py   (locate -> ablate -> evaluate)")
    print("=" * 60)


if __name__ == "__main__":
    main()
