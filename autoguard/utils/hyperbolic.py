#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hyperbolic geometry utilities for Poincaré ball operations.
"""

import torch
import math


def poincare_ball_project(x: torch.Tensor, curvature: float = 1.0,
                          eps: float = 1e-5) -> torch.Tensor:
    """Project points to inside the Poincaré ball.

    Args:
        x: Points to project [*, dim]
        curvature: Ball curvature (positive)
        eps: Margin from boundary

    Returns:
        Projected points guaranteed inside the ball
    """
    max_norm = (1.0 - eps) / math.sqrt(curvature)
    norms = torch.norm(x, dim=-1, keepdim=True)
    projected = torch.where(
        norms > max_norm,
        x * max_norm / norms,
        x
    )
    return projected


def hyperbolic_distance(u: torch.Tensor, v: torch.Tensor,
                        curvature: float = 1.0) -> torch.Tensor:
    """Compute pairwise hyperbolic distance in Poincaré ball.

    Args:
        u, v: Points in Poincaré ball [batch, dim]
        curvature: Ball curvature

    Returns:
        Distances [batch, 1]
    """
    c = curvature
    sqrt_c = math.sqrt(c)

    diff = u - v
    diff_norm_sq = torch.sum(diff * diff, dim=-1, keepdim=True).clamp(min=1e-10)
    u_norm_sq = torch.sum(u * u, dim=-1, keepdim=True).clamp(max=1.0 / c - 1e-5)
    v_norm_sq = torch.sum(v * v, dim=-1, keepdim=True).clamp(max=1.0 / c - 1e-5)

    num = 2 * c * diff_norm_sq
    denom = (1 - c * u_norm_sq) * (1 - c * v_norm_sq)
    denom = denom.clamp(min=1e-10)

    arg = 1 + num / denom
    dist = (1.0 / sqrt_c) * torch.acosh(arg.clamp(min=1.0 + 1e-7))

    return dist


def mobius_addition(u: torch.Tensor, v: torch.Tensor,
                    curvature: float = 1.0) -> torch.Tensor:
    """Mobius addition in Poincaré ball.

    Args:
        u, v: Points/vectors [batch, dim]
        curvature: Ball curvature

    Returns:
        Result of Möbius addition [batch, dim]
    """
    c = curvature
    u_norm_sq = torch.sum(u * u, dim=-1, keepdim=True)
    v_norm_sq = torch.sum(v * v, dim=-1, keepdim=True)
    uv = torch.sum(u * v, dim=-1, keepdim=True)

    num = (1 + 2 * c * uv + c * v_norm_sq) * u + (1 - c * u_norm_sq) * v
    denom = 1 + 2 * c * uv + c * c * u_norm_sq * v_norm_sq
    denom = denom.clamp(min=1e-10)

    result = num / denom
    return poincare_ball_project(result, curvature)


def exponential_map(x: torch.Tensor, v: torch.Tensor,
                    curvature: float = 1.0) -> torch.Tensor:
    """Exponential map from tangent space at x to Poincaré ball.

    Maps tangent vector v at point x to the manifold.

    Args:
        x: Base point [batch, dim]
        v: Tangent vector [batch, dim]
        curvature: Ball curvature

    Returns:
        Point on manifold [batch, dim]
    """
    c = curvature
    sqrt_c = math.sqrt(c)
    v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-10)
    x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)

    lambda_x = 2.0 / (1 - c * x_norm_sq).clamp(min=1e-10)
    second_term = torch.tanh(sqrt_c * lambda_x * v_norm / 2) * v / (sqrt_c * v_norm)

    return mobius_addition(x, second_term, curvature)


def logarithmic_map(x: torch.Tensor, y: torch.Tensor,
                    curvature: float = 1.0) -> torch.Tensor:
    """Logarithmic map from Poincaré ball to tangent space at x.

    Inverse of exponential map.

    Args:
        x: Base point [batch, dim]
        y: Target point [batch, dim]
        curvature: Ball curvature

    Returns:
        Tangent vector at x pointing toward y [batch, dim]
    """
    c = curvature
    sqrt_c = math.sqrt(c)

    # Mobius addition: -x (+) y
    neg_x = -x
    diff = mobius_addition(neg_x, y, curvature)
    diff_norm = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-10)

    x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
    lambda_x = 2.0 / (1 - c * x_norm_sq).clamp(min=1e-10)

    factor = (2.0 / (sqrt_c * lambda_x)) * torch.atanh(sqrt_c * diff_norm)
    return factor * diff / diff_norm


def parallel_transport(x: torch.Tensor, y: torch.Tensor,
                       v: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    """Parallel transport of vector v from tangent space at x to tangent space at y.

    Args:
        x: Source point [batch, dim]
        y: Target point [batch, dim]
        v: Vector to transport [batch, dim]
        curvature: Ball curvature

    Returns:
        Transported vector in tangent space at y [batch, dim]
    """
    c = curvature
    x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
    y_norm_sq = torch.sum(y * y, dim=-1, keepdim=True)

    lambda_x = 2.0 / (1 - c * x_norm_sq).clamp(min=1e-10)
    lambda_y = 2.0 / (1 - c * y_norm_sq).clamp(min=1e-10)

    return v * (lambda_x / lambda_y)
