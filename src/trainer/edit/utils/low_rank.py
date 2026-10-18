"""Exact low-rank linear solves used by scalable edit updates."""

from __future__ import annotations

import math

import torch


def _validate(features: torch.Tensor, rhs: torch.Tensor, ridge: float) -> None:
    if features.ndim != 2 or rhs.ndim != 2:
        raise ValueError("features and rhs must both be rank-2 tensors")
    if features.shape[0] != rhs.shape[0]:
        raise ValueError("features and rhs must share their leading dimension")
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be positive and finite")


def solve_ridge_low_rank(
    features: torch.Tensor,
    rhs: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    """Solve ``(ridge I + F F^T) X = rhs`` via a small Woodbury system.

    Accumulate in double precision: for a small ridge and rhs in the feature
    span, the final subtraction cancels most significant digits in float32.
    Return the original rhs dtype without forming an input-dimension matrix.
    """
    _validate(features, rhs, ridge)
    output_dtype = rhs.dtype
    features, rhs = features.double(), rhs.double()
    if features.shape[1] == 0:
        return (rhs / ridge).to(output_dtype)

    small = features.T @ features
    small.diagonal().add_(ridge)
    coefficients = torch.linalg.solve(small, features.T @ rhs)
    return ((rhs - features @ coefficients) / ridge).to(output_dtype)


def solve_projected_ridge_low_rank(
    projected_features: torch.Tensor,
    features: torch.Tensor,
    rhs: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    """Solve ``(ridge I + P F F^T) X = rhs`` when ``P F`` is provided."""
    _validate(features, rhs, ridge)
    if projected_features.shape != features.shape:
        raise ValueError("projected_features must have the same shape as features")
    # The projected formula has the same subtractive cancellation as above.
    output_dtype = rhs.dtype
    projected_features, features, rhs = (
        projected_features.double(),
        features.double(),
        rhs.double(),
    )
    if features.shape[1] == 0:
        return (rhs / ridge).to(output_dtype)

    small = features.T @ projected_features
    small.diagonal().add_(ridge)
    coefficients = torch.linalg.solve(small, features.T @ rhs)
    return ((rhs - projected_features @ coefficients) / ridge).to(output_dtype)


__all__ = ["solve_projected_ridge_low_rank", "solve_ridge_low_rank"]
