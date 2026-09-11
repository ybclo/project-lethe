"""
Project Lethe - core/ablate.py

The Erasure Engine performs **Targeted Concept Ablation**: a surgical edit
that removes the target concept from the model's weights while leaving
general capabilities intact.

Method (orthogonal projection / task-vector negation):

    Given a unit Concept Vector `d` (the direction in the residual stream
    that encodes the concept), each target weight matrix `W` is replaced by

        W' = W - strength * (W @ d) @ d^T        ==    W @ (I - strength * d d^T)

    For `strength = 1.0` the factor (I - d d^T) is the orthogonal projector
    onto the subspace perpendicular to `d`, so after the edit:

        W' @ d = W @ d - (W @ d) (d^T d) = 0

    i.e. the edited matrix no longer carries any component along the concept
    direction. `strength` scales how aggressively the direction is removed.

We apply the edit to the MLP output projection (`mlp.c_proj`) of the target
blocks, which is where concepts are most directly written into the residual
stream.

The engine always works on a *deep copy*, so the original model is left
untouched (this is what lets us run before/after comparisons).
"""

from __future__ import annotations

import copy

import torch


class ConceptAblator:
    """Applies orthogonal-projection ablation for a single concept vector."""

    def __init__(self, model, concept_vector, target_layers=None, ablation_strength=1.0):
        self.source_model = model
        self.model = copy.deepcopy(model)  # never mutate the input model
        self.model.eval()

        v = torch.as_tensor(concept_vector, dtype=torch.float32).reshape(-1)
        n = v.norm()
        self.concept_vector = (v / n if n > 0 else v).to(self.model.device)

        self.num_layers = len(self.model.transformer.h)
        self.target_layers = (
            list(range(self.num_layers)) if target_layers is None else list(target_layers)
        )
        self.ablation_strength = float(ablation_strength)

        # Keep the pre-edit weights of the targeted layers for reference.
        self.original_weights = {}
        for idx in self.target_layers:
            self.original_weights[idx] = (
                self.model.transformer.h[idx].mlp.c_proj.weight.data.clone()
            )

    # ------------------------------------------------------------------ #
    def apply_ablation(self, layer_idx):
        """
        Project one layer's MLP output weights orthogonal to the concept.

            W' = W - strength * (W @ d) @ d^T

        Returns the ablator for chaining.
        """
        block = self.model.transformer.h[layer_idx].mlp
        proj = getattr(block, "c_proj", None)
        if proj is None:
            return self

        W = proj.weight.data.to(torch.float32)
        d = self.concept_vector  # unit vector, [hidden_dim]

        Wd = W @ d                       # [out_features]
        proj_update = Wd.unsqueeze(1) @ d.unsqueeze(0)  # [out, in]
        proj.weight.data = (W - self.ablation_strength * proj_update).to(
            proj.weight.dtype
        )
        return self

    def ablate_all_layers(self):
        """
        Apply the projection edit to every target layer.

        Returns the ablator itself so callers can chain:
            ablator.ablate_all_layers().get_ablated_model()
        """
        for idx in self.target_layers:
            self.apply_ablation(idx)
        return self

    def get_ablated_model(self):
        """Return the (edited) deep copy of the model."""
        return self.model

    def weight_change_stats(self):
        """
        Report how much the edit moved the weights, per targeted layer.

        Returns:
            dict layer_idx -> {"mean_abs_change": float, "max_abs_change": float}
        """
        stats = {}
        for idx in self.target_layers:
            before = self.original_weights[idx]
            after = self.model.transformer.h[idx].mlp.c_proj.weight.data.to(before.dtype)
            delta = (after - before).abs()
            stats[idx] = {
                "mean_abs_change": float(delta.mean().item()),
                "max_abs_change": float(delta.max().item()),
            }
        return stats


def ablation_weights(model, concept_vector, target_layers=None, ablation_strength=1.0):
    """
    Convenience wrapper: apply concept ablation and return the edited model.

    Args:
        model:             HuggingFace causal LM (left unmodified)
        concept_vector:    tensor [hidden_dim] concept direction
        target_layers:     list of block indices (default: all)
        ablation_strength: 1.0 = full orthogonal projection

    Returns:
        GPT2LMHeadModel (deep copy) with the concept projected out.
    """
    ablator = ConceptAblator(
        model,
        concept_vector,
        target_layers=target_layers,
        ablation_strength=ablation_strength,
    )
    ablator.ablate_all_layers()
    return ablator.get_ablated_model()


if __name__ == "__main__":
    print("Project Lethe - ConceptAblator self-test")

    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    hidden = model.config.n_embd
    rng = torch.Generator().manual_seed(0)
    concept_vector = torch.randn(hidden, generator=rng)
    concept_vector = concept_vector / concept_vector.norm()

    # Snapshot the source weights so we can prove the source is untouched.
    source_before = [
        h.mlp.c_proj.weight.data.clone() for h in model.transformer.h
    ]

    edited = ablation_weights(model, concept_vector, ablation_strength=1.0)

    d = concept_vector
    worst, source_worst = 0.0, 0.0
    for i in range(len(edited.transformer.h)):
        Wp = edited.transformer.h[i].mlp.c_proj.weight.data
        worst = max(worst, float((Wp @ d).abs().max().item()))
        source_worst = max(source_worst, float((source_before[i] @ d).abs().max().item()))
    untouched = all(torch.equal(w, b) for w, b in
                    zip((h.mlp.c_proj.weight.data for h in model.transformer.h),
                        source_before))
    print("  source max |W @ d| : {:.3e} (non-zero = had a concept component)".format(source_worst))
    print("  edited max |W' @ d|: {:.3e} (should be ~0)".format(worst))
    print("  source weights untouched: {}".format(untouched))
    assert worst < 1e-4, "projection identity failed"
    assert source_worst > 0.0, "source had no component along d"
    assert untouched, "source model was mutated"
    print("  ablation self-test passed.")
