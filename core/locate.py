"""
Project Lethe - core/locate.py

The Location Engine identifies which neurons "light up" when the model
thinks about a specific concept, and distills that into a single
**Concept Vector**: the direction in the residual stream that represents
the target concept.

Method (activation tracing):
    1. Generate many prompts about the concept.
    2. Run each prompt through the model and capture the mean MLP output
       (over the sequence axis) for every transformer block, via forward
       hooks.
    3. Average each layer's activation across all prompts -> per-layer
       mean activation vectors.
    4. Combine the per-layer vectors (later layers weighted slightly more,
       since they carry higher-level semantics) and normalise to unit
       length.

The resulting unit vector `d` is what core/ablate.py projects out of the
weights.

Notes:
    - Works for any causal LM whose blocks are at `model.transformer.h`
      with an `.mlp.c_proj` projection (GPT-2 family).
    - Device and dimensions are read from the model config, so it is not
      hard-wired to a specific model size.
"""

from __future__ import annotations

import torch
from torch import nn


class ConceptLocator:
    """Locates the Concept Vector for a target concept via activation tracing."""

    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # GPT-2 family has no pad token by default; hooks need a defined one.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.to(self.device)
        self.model.eval()

        # Per-run storage.
        self._handles = []
        self._activations = {}  # layer_name -> tensor [1, hidden_dim]

        # Geometry, derived from the model (no hard-coded sizes).
        self.num_layers = len(model.transformer.h)
        self.hidden_dim = model.config.n_embd
        self._layer_names = [f"transformer.h.{i}.mlp" for i in range(self.num_layers)]

    # ------------------------------------------------------------------ #
    # Hook management
    # ------------------------------------------------------------------ #
    def _make_hook(self, layer_name):
        def hook(module, inp, output):
            # For GPT-2 the MLP returns a plain tensor: [batch, seq, hidden].
            hidden = output if not isinstance(output, (tuple, list)) else output[0]
            # Average over the sequence axis -> [batch, hidden] (length-invariant).
            self._activations[layer_name] = hidden.detach().float().mean(dim=1)
        return hook

    def _register_hooks(self):
        for name in self._layer_names:
            i = int(name.split(".")[2])
            block = self.model.transformer.h[i]
            self._handles.append(block.mlp.register_forward_hook(self._make_hook(name)))

    def _remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    # ------------------------------------------------------------------ #
    # Prompt generation
    # ------------------------------------------------------------------ #
    def generate_concept_prompts(self, concept_texts, num_prompts=500):
        """Expand a few seed statements into many concept-probing prompts."""
        templates = [
            "Tell me about {c}.",
            "Who is {c}?",
            "{c} story",
            "{c} biography",
            "What do you know about {c}?",
            "Describe {c}.",
            "{c} facts",
            "The character {c} appears in",
        ]
        adjectives = ["fictional", "famous", "notable", "popular", "classic", "modern"]

        seeds = list(concept_texts)[: max(1, min(10, len(concept_texts)))]
        prompts = []
        for text in seeds:
            for t in templates:
                if len(prompts) >= num_prompts:
                    return prompts[:num_prompts]
                prompts.append(t.format(c=text))
        for adj in adjectives:
            for text in seeds[:5]:
                if len(prompts) >= num_prompts:
                    return prompts[:num_prompts]
                prompts.append("The {} story of {}.".format(adj, text))
        return prompts[:num_prompts] if prompts else list(concept_texts)

    # ------------------------------------------------------------------ #
    # Activation tracing
    # ------------------------------------------------------------------ #
    def trace_activations(self, prompts, max_length=64):
        """
        Run prompts through the model and capture per-layer mean activations.

        Returns:
            dict: layer_name -> mean activation tensor of shape [hidden_dim]
        """
        per_prompt = {name: [] for name in self._layer_names}

        self._register_hooks()
        try:
            with torch.no_grad():
                for prompt in prompts:
                    enc = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length,
                    ).to(self.device)
                    _ = self.model(**enc)
                    for name in self._layer_names:
                        act = self._activations.get(name)
                        if act is not None:
                            per_prompt[name].append(act.reshape(-1).clone())
        finally:
            self._remove_hooks()
            self._activations = {}

        layer_means = {}
        for name in self._layer_names:
            acts = per_prompt[name]
            if acts:
                layer_means[name] = torch.stack(acts, dim=0).mean(dim=0)
        return layer_means

    # ------------------------------------------------------------------ #
    # Concept vector
    # ------------------------------------------------------------------ #
    def compute_concept_vector(self, concept_texts, num_prompts=500):
        """
        Compute the unit-length Concept Vector for a target concept.

        Returns:
            (concept_vector, layer_means):
                concept_vector: tensor [hidden_dim], norm 1
                layer_means:    dict layer_name -> tensor [hidden_dim]
        """
        prompts = self.generate_concept_prompts(concept_texts, num_prompts)
        if not prompts:
            prompts = list(concept_texts)

        layer_means = self.trace_activations(prompts)

        concept_vector = self._combine_layer_vectors(layer_means)
        norm = concept_vector.norm()
        if norm > 0:
            concept_vector = concept_vector / norm
        return concept_vector, layer_means

    def _combine_layer_vectors(self, layer_means):
        """Weighted average of per-layer vectors; later layers weigh more."""
        present = [(i, layer_means[f"transformer.h.{i}.mlp"])
                   for i in range(self.num_layers)
                   if f"transformer.h.{i}.mlp" in layer_means]
        if not present:
            return torch.zeros(self.hidden_dim, device=self.device)

        total_w = 0.0
        acc = torch.zeros(self.hidden_dim, device=self.device)
        for i, vec in present:
            w = (i + 1) / self.num_layers  # later layers matter more
            acc = acc + w * vec
            total_w += w
        return acc / total_w if total_w > 0 else acc

    def identify_target_layers(self, concept_vector, threshold=0.1):
        """
        Rank layers by how strongly their MLP output projection aligns with
        the concept vector. Useful for focusing the ablation.

        Returns:
            list of (layer_index, importance) sorted descending.
        """
        scores = []
        d = concept_vector.reshape(-1)
        for i in range(self.num_layers):
            block = self.model.transformer.h[i]
            proj = getattr(block.mlp, "c_proj", None)
            if proj is None:
                continue
            W = proj.weight.data
            try:
                # Row-wise alignment of the output projection with d.
                scores.append((i, float(torch.abs(W @ d).mean().item())))
            except RuntimeError:
                continue
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


def locate_concept_vector(model, tokenizer, concept_texts, num_prompts=500, device=None):
    """
    Convenience wrapper: locate a concept vector for the given seed texts.

    Returns:
        (concept_vector, locator)
    """
    locator = ConceptLocator(model, tokenizer, device=device)
    concept_vector, _ = locator.compute_concept_vector(concept_texts, num_prompts=num_prompts)
    return concept_vector, locator
