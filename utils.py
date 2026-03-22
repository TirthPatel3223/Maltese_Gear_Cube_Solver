"""
utils.py  --  Utilities for Maltese Gear Cube Training

Contains:
  - AMP helpers (autocast, GradScaler)
  - Hardware profiling for batch-size selection
  - HL-Gauss soft-target computation (Farebrother et al., 2024)
  - Curriculum-scaled average-KL sync threshold  (get_avg_kl_sync_threshold)
  - Cumulative GBFS progress plotting
  - Logging
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# AMP Helpers
# ============================================================================

def autocast_ctx(device):
    """Context manager for automatic mixed precision on CUDA."""
    return torch.amp.autocast("cuda", enabled=(device.type == "cuda"))


def make_scaler(device):
    """Create GradScaler for mixed-precision training."""
    return torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))


# ============================================================================
# Hardware Profiling
# ============================================================================

def get_hw_profile(device):
    """
    Select batch sizes based on available GPU memory.

    Returns dict with TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE,
    CHILD_EVAL_CHUNK, and SCRAMBLE_CHUNK.
    """
    if device.type != "cuda":
        return {
            "TRAIN_BATCH_SIZE": 4096,
            "EVAL_BATCH_SIZE":  4096,
            "CHILD_EVAL_CHUNK": 25_000,
            "SCRAMBLE_CHUNK":   200_000,
        }

    total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)

    if total_gb <= 16:
        print(f"Detected GPU with {total_gb:.1f}GB VRAM -> Using 12GB Profile.")
        return {
            "TRAIN_BATCH_SIZE": 40_000,
            "EVAL_BATCH_SIZE":  12_000,
            "CHILD_EVAL_CHUNK": 50_000,
            "SCRAMBLE_CHUNK":   1_000_000,
        }

    print(f"Detected GPU with {total_gb:.1f}GB VRAM -> Using 40GB+ A100 Profile.")
    return {
        "TRAIN_BATCH_SIZE": 100_000,
        "EVAL_BATCH_SIZE":  45_000,
        "CHILD_EVAL_CHUNK": 200_000,
        "SCRAMBLE_CHUNK":   2_000_000,
    }


# ============================================================================
# HL-Gauss Soft Targets
# ============================================================================

def create_soft_targets(targets_float, max_dist, device, smoothing_type="hl_gauss", sigma=0.75):
    """
    Convert scalar Bellman targets to HL-Gauss soft probability distributions.

    Implements equation 3.3 from Farebrother et al. (2024), "Stop Regressing:
    Training Value Functions via Classification," ICML 2024.

    Uses boundary correction (mass piling) to ensure the distribution sums
    to exactly 1.0 without distortion.

    Args:
        targets_float : (B,) float tensor of scalar targets
        max_dist      : number of bins (support size)
        device        : torch device
        sigma         : Gaussian smoothing parameter; sigma/zeta=0.75 recommended

    Returns:
        (B, max_dist) float tensor of soft target probabilities
    """
    support          = torch.arange(0, max_dist, device=device).float().unsqueeze(0)
    targets_expanded = targets_float.unsqueeze(1)
    sqrt2_sigma      = (2.0 ** 0.5) * sigma

    cdf_left = 0.5 * (1.0 + torch.special.erf(
        (support - 0.5 - targets_expanded) / sqrt2_sigma
    ))
    cdf_right = 0.5 * (1.0 + torch.special.erf(
        (support + 0.5 - targets_expanded) / sqrt2_sigma
    ))

    target_probs = cdf_right - cdf_left

    # Boundary correction: pile tail mass onto edge bins
    left_tail = 0.5 * (1.0 + torch.special.erf(
        (-0.5 - targets_float) / sqrt2_sigma
    ))
    target_probs[:, 0] += left_tail

    right_tail = 1.0 - 0.5 * (1.0 + torch.special.erf(
        (max_dist - 0.5 - targets_float) / sqrt2_sigma
    ))
    target_probs[:, -1] += right_tail

    return target_probs


# ============================================================================
# Average-KL Sync Threshold  (DeepCubeA-style, curriculum-scaled)
# ============================================================================

def get_avg_kl_sync_threshold(back_max, base_threshold=0.05, first_stage_max=30, scale_power=0.5):
    """
    Curriculum-scaled average-KL threshold for target-network sync.

    At back_max == first_stage_max the threshold equals base_threshold.
    For deeper curriculum stages the threshold grows as a power law of
    (back_max / first_stage_max), reflecting the larger, harder state space.

    Formula:
        threshold = base_threshold * (back_max / first_stage_max) ^ scale_power

    Default parameters (base_threshold=0.05, first_stage_max=30, scale_power=0.5):
        back_max=30  ->  0.050
        back_max=60  ->  0.071
        back_max=120 ->  0.100
        back_max=200 ->  0.129
        back_max=500 ->  0.204

    Args:
        back_max        : current curriculum scramble depth
        base_threshold  : sync threshold at the first curriculum stage
        first_stage_max : back_max value of the first curriculum stage
        scale_power     : exponent of the power-law scaling (0.5 = sqrt)

    Returns:
        float threshold value
    """
    return base_threshold * (back_max / first_stage_max) ** scale_power


# ============================================================================
# GBFS History -- Cumulative Plotting
# ============================================================================

def load_gbfs_history(filepath="saved_models/gbfs_history.json"):
    """Load accumulated GBFS evaluation history from disk."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return []


def save_gbfs_history(history, filepath="saved_models/gbfs_history.json"):
    """Save accumulated GBFS evaluation history to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(history, f, indent=2)


def append_gbfs_result(gbfs_res, itr, back_max, filepath="saved_models/gbfs_history.json"):
    """
    Append a single GBFS evaluation result to the cumulative history.

    Each entry records the evaluation iteration, curriculum depth,
    per-depth solve percentages, CTG predictions, and overestimation rates.
    """
    history = load_gbfs_history(filepath)
    entry = {
        "itr":       itr,
        "back_max":  back_max,
        "depths":    gbfs_res["depths"],
        "pct_solved": gbfs_res["pct_solved"],
        "avg_ctg":   gbfs_res["avg_ctg"],
    }
    if "pct_overestimated" in gbfs_res:
        entry["pct_overestimated"] = gbfs_res["pct_overestimated"]
    history.append(entry)
    save_gbfs_history(history, filepath)
    return history


def plot_training_progress(filepath="saved_models/gbfs_history.json",
                           output="saved_models/training_progress.png"):
    """
    Plot cumulative GBFS solve rates across all training iterations.

    Creates a heatmap-style plot showing solve percentage evolution
    across curriculum stages, and a summary curve of overall solve rate.
    """
    history = load_gbfs_history(filepath)
    if not history:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax1    = axes[0]
    cmap   = plt.cm.viridis
    n_evals = len(history)

    for idx, entry in enumerate(history):
        color = cmap(idx / max(1, n_evals - 1))
        label = (
            f"itr={entry['itr']}, d={entry['back_max']}"
            if idx % max(1, n_evals // 8) == 0
            else None
        )
        ax1.plot(
            entry["depths"], entry["pct_solved"],
            color=color, alpha=0.7, linewidth=1.5, label=label,
        )

    ax1.set_xlabel("Scramble Depth")
    ax1.set_ylabel("% Solved (GBFS)")
    ax1.set_ylim(-5, 105)
    ax1.set_title("GBFS Solve Rate Across Training Iterations")
    if n_evals <= 16:
        ax1.legend(fontsize=7, loc="lower left")
    ax1.grid(True, alpha=0.3)

    ax2          = axes[1]
    itrs         = [e["itr"]                    for e in history]
    overall_solve = [np.mean(e["pct_solved"])   for e in history]
    max_depths   = [e["back_max"]               for e in history]

    color_solve = "tab:blue"
    ax2.plot(itrs, overall_solve, color=color_solve, marker="o",
             markersize=4, linewidth=2, label="Mean Solve %")
    ax2.set_xlabel("Training Iteration")
    ax2.set_ylabel("Mean Solve % (all depths)", color=color_solve)
    ax2.tick_params(axis="y", labelcolor=color_solve)
    ax2.set_ylim(-5, 105)

    ax2_twin = ax2.twinx()
    color_depth = "tab:red"
    ax2_twin.plot(itrs, max_depths, color=color_depth, marker="s",
                  markersize=4, linewidth=2, linestyle="--", label="Curriculum Depth")
    ax2_twin.set_ylabel("Curriculum back_max", color=color_depth)
    ax2_twin.tick_params(axis="y", labelcolor=color_depth)

    ax2.set_title("Training Progress Summary")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="center left")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Training progress plot saved to {output}")


# ============================================================================
# Logging
# ============================================================================

class Logger:
    """Tee stdout to both terminal and a log file."""

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log      = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
