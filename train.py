"""
train.py  --  Multi-Epoch DAVI · DeepCubeA-Style Training-KL Target Sync

Architecture changes vs. previous version:
  1. all_zeros DeepCubeA initialization (Agostinelli et al., 2019, avi.py L208)
  2. Stagnation-based forced sync replaces fixed max_update_cycles counter
  3. Training KL used directly for sync decision (no separate validation)

Initialization (all_zeros):
    On a fresh training run, the target network's random weights are NEVER
    queried.  Instead, every child state is treated as having CTG = 0, so:
        target(s) = 0   if s is solved
        target(s) = 1   otherwise
    This provides a perfectly clean first training signal: "every non-goal
    state is at least one step away from solved."  This is the correct
    initial condition for value iteration starting from J_0 = 0.

    all_zeros stays True for EVERY cycle until the first KL- or stagnation-
    triggered sync fires.  After that sync, all_zeros becomes False
    permanently (even across curriculum stages), because the trained
    target_net is always more informative than zeros.

    References:
        Agostinelli et al. (2019) — DeepCubeA, avi.py line 208
        Bertsekas (2001) — Neuro-Dynamic Programming §6.3, Bellman contraction

Sync philosophy:
    After every update cycle, the final-epoch average training KL is used
    to decide whether to sync target_net <- current_net.

    KL-triggered sync  : if train_kl < kl_threshold → sync.
    Stagnation sync    : if relative KL improvement over the last
                         kl_stagnation_patience cycles < kl_stagnation_min_delta
                         → sync.  The target has become stale; refreshing it
                         allows Bellman contraction to resume.
    On sync            : kl_history clears, use_all_zeros becomes False.

Threshold schedule (power-law, DeepCubeA-inspired):
    kl_threshold = kl_base * (back_max / curriculum[0]) ^ kl_scale_power

Curriculum advancement : purely GBFS-gated (unchanged).
max_stage_cycles safety cap : unchanged.

VRAM hardening:
    - NEITHER network uses CUDAGraphs (mode="max-autotune-no-cudagraphs")
    - target_net is not compiled at all (eager mode under no_grad)
    - Pre-allocated ctg_buf eliminates torch.empty inside Bellman loop
    - In-place operations replace torch.where for solved masks
    - gc.collect() + synchronize + empty_cache at cycle boundaries
"""

import argparse
import collections
import gc
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from environment import MalteseGearCubeEnv
from model import CategoricalResNet
from search import test_gbfs_gpu
from utils import (
    Logger,
    append_gbfs_result,
    autocast_ctx,
    create_soft_targets,
    get_avg_kl_sync_threshold,
    get_hw_profile,
    make_scaler,
    plot_training_progress,
)


# ============================================================================
# Argparse
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a CategoricalResNet to solve the Maltese Gear Cube "
            "using DAVI with DeepCubeA-style training-KL target sync."
        )
    )

    # -- Training budget ------------------------------------------------------
    parser.add_argument(
        "--max_itrs", type=int, default=1_000_000,
        help="Total gradient steps across all training (default: 1 000 000)",
    )
    parser.add_argument(
        "--states_per_update", type=int, default=15_000_000,
        help="Training states generated per update cycle (default: 12 000 000)",
    )

    # -- Learning rate --------------------------------------------------------
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Initial learning rate for Adam (default: 0.001)",
    )
    parser.add_argument(
        "--lr_decay", type=float, default=0.9999998,
        help="Per-step multiplicative LR decay (default: 0.9999998)",
    )

    # -- Model architecture ---------------------------------------------------
    parser.add_argument(
        "--max_dist", type=int, default=500,
        help="Categorical support size / number of bins (default: 505)",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=7000,
        help="First FC layer width (default: 5000)",
    )
    parser.add_argument(
        "--res_dim", type=int, default=2000,
        help="Residual block width (default: 1000)",
    )
    parser.add_argument(
        "--num_res_blocks", type=int, default=6,
        help="Number of residual blocks (default: 6)",
    )

    # -- Curriculum -----------------------------------------------------------
    parser.add_argument(
        "--curriculum", type=int, nargs="+",
        default=[30, 60, 120, 200, 500],
        help="Scramble-depth schedule (default: 30 60 120 200 500)",
    )

    # -- Inner training loop --------------------------------------------------
    parser.add_argument(
        "--max_inner_epochs", type=int, default=1,
        help="Max epochs per update cycle on fixed Bellman targets (default: 1)",
    )
    parser.add_argument(
        "--early_stop_patience", type=int, default=2,
        help="Inner-loop early-stop patience in epochs (default: 2)",
    )
    parser.add_argument(
        "--min_improvement", type=float, default=0.01,
        help="Min relative loss improvement to reset patience (default: 0.01)",
    )

    # -- KL-triggered target sync ---------------------------------------------
    parser.add_argument(
        "--kl_base", type=float, default=0.05,
        help=(
            "Avg-KL sync threshold at curriculum[0].  "
            "Full formula: kl_base*(back_max/curriculum[0])^kl_scale_power. "
            "(default: 0.05)"
        ),
    )
    parser.add_argument(
        "--kl_scale_power", type=float, default=0.5,
        help="Power-law exponent for threshold scaling with back_max (default: 0.5 = sqrt)",
    )

    # -- Stagnation-based forced sync -----------------------------------------
    parser.add_argument(
        "--kl_stagnation_patience", type=int, default=10,
        help=(
            "Number of consecutive update cycles tracked for stagnation "
            "detection.  If the relative improvement in training KL from "
            "the oldest to newest entry in this window is below "
            "--kl_stagnation_min_delta, a forced sync fires. (default: 10)"
        ),
    )
    parser.add_argument(
        "--kl_stagnation_min_delta", type=float, default=0.016,
        help=(
            "Minimum relative KL improvement over the stagnation window "
            "before forced sync fires.  0.01 = 1%%. (default: 0.01)"
        ),
    )

    # -- Stage safety cap -----------------------------------------------------
    parser.add_argument(
        "--max_stage_cycles", type=int, default=1000,
        help=(
            "Maximum total update cycles allowed per curriculum stage. "
            "If exhausted without GBFS advancement the stage restarts from "
            "cycle 0. (default: 1000)"
        ),
    )

    # -- GBFS evaluation ------------------------------------------------------
    parser.add_argument(
        "--gbfs_eval_freq", type=int, default=5,
        help="Run GBFS every N update cycles (default: 5)",
    )
    parser.add_argument(
        "--gbfs_num_test", type=int, default=2500,
        help="Total states used for GBFS evaluation (default: 2500)",
    )
    parser.add_argument(
        "--gbfs_solve_threshold", type=float, default=75.0,
        help="Min GBFS solve %% per depth bin for curriculum advancement (default: 75)",
    )
    parser.add_argument(
        "--min_solve_depth_fraction", type=float, default=0.60,
        help="Fraction of depth bins that must pass GBFS gate (default: 0.50)",
    )

    # -- HL-Gauss -------------------------------------------------------------
    parser.add_argument(
        "--sigma", type=float, default=0.75,
        help="HL-Gauss smoothing sigma (default: 0.75)",
    )

    # -- Gradient clipping ----------------------------------------------------
    parser.add_argument(
        "--grad_clip", type=float, default=1.0,
        help="Max gradient norm for clipping (default: 1.0; 0 = disabled)",
    )

    # -- Checkpointing --------------------------------------------------------
    parser.add_argument(
        "--save_dir", type=str, default="saved_models",
        help="Directory for checkpoints and logs (default: saved_models)",
    )

    # -- Misc -----------------------------------------------------------------
    parser.add_argument(
        "--no_compile", action="store_true",
        help="Disable torch.compile even if available",
    )

    return parser.parse_args()


# ============================================================================
# Bellman Target Computation
# ============================================================================

def compute_bellman_targets_gpu(
    env,
    target_net,
    states_tensor,
    batch_size,
    device,
    child_eval_chunk,
    ctg_buf,
    all_zeros=False,
    verbose=True,
):
    """
    Compute one-step Bellman targets.

        target(s) = 0                             if s is solved
        target(s) = 1 + min_a J_target(child_a)  otherwise

    all_zeros=True  (DeepCubeA initialization mode):
        Treats ALL child states as having J_target = 0, bypassing the network
        entirely.  Result: target(s) = 0 if solved, else 1.
        Active on EVERY cycle of a fresh run until the first sync fires.

    all_zeros=False (normal DAVI mode):
        Frozen target_net evaluates children.  target_net is eager (not
        compiled) since compilation adds no benefit under torch.no_grad().

    VRAM design:
        ctg_buf : pre-allocated [batch_size * env.num_moves] float32 tensor.
            Each outer batch takes a VIEW (slice) into it — no new allocation.
        In-place operations replace torch.where for solved masks.

    References:
        Agostinelli et al. (2019) — DeepCubeA, avi.py line 208
        Bertsekas (2001) — Bellman contraction theorem §6.3
    """
    if verbose:
        if all_zeros:
            print(
                "Computing Bellman targets [ALL-ZEROS mode — "
                "target = 0 if solved, else 1]..."
            )
        else:
            print("Computing Bellman targets with frozen target network (eager)...")

    start_time  = time.time()
    num_states  = states_tensor.size(0)
    targets     = torch.empty(num_states, dtype=torch.float32, device=device)

    if all_zeros:
        # -----------------------------------------------------------------
        # DeepCubeA-exact initialization: every child CTG treated as 0.
        # No network forward pass needed — just check if parent is solved.
        # -----------------------------------------------------------------
        with torch.no_grad():
            for i in range(0, num_states, batch_size):
                end_idx      = min(i + batch_size, num_states)
                batch_states = states_tensor[i:end_idx]
                solved_mask  = env.is_solved_gpu(batch_states)
                targets[i:end_idx] = torch.where(solved_mask, 0.0, 1.0)

    else:
        # -----------------------------------------------------------------
        # Normal DAVI: evaluate children with frozen target network.
        # -----------------------------------------------------------------
        target_net.eval()

        with torch.no_grad():
            for i in range(0, num_states, batch_size):
                end_idx      = min(i + batch_size, num_states)
                batch_states = states_tensor[i:end_idx]
                actual_batch = batch_states.size(0)

                children      = env.expand_gpu(batch_states)
                children_flat = children.view(-1, 144)
                num_children  = children_flat.size(0)

                # View into pre-allocated buffer (no new allocation)
                ctg_next_flat = ctg_buf[:num_children]

                for j in range(0, num_children, child_eval_chunk):
                    chunk       = children_flat[j : j + child_eval_chunk]
                    actual_size = chunk.size(0)

                    # target_net is eager — varying sizes are fine, no padding
                    inputs = env.states_to_nnet_input_static(chunk)

                    with autocast_ctx(device):
                        logits = target_net(inputs)

                    ctg_next_flat[j : j + actual_size] = (
                        target_net.get_ctg(logits)[:actual_size]
                    )

                    del logits  # explicit cleanup of intermediate

                # Solved children → CTG = 0 (in-place)
                solved_children_mask = env.is_solved_gpu(children_flat)
                ctg_next_flat[solved_children_mask] = 0.0

                # Reshape and take min across moves
                ctg_next    = ctg_next_flat[:num_children].view(actual_batch, env.num_moves)
                best_ctg, _ = torch.min(ctg_next, dim=1)

                # target = best_ctg + 1, but solved parents → 0 (in-place)
                best_ctg.add_(1.0)
                solved_parent_mask = env.is_solved_gpu(batch_states)
                best_ctg[solved_parent_mask] = 0.0

                targets[i:end_idx] = best_ctg

                # Free intermediates from this outer batch
                del children, children_flat, solved_children_mask
                del ctg_next, solved_parent_mask, best_ctg

    if verbose:
        print(f"Bellman targets computed in {time.time() - start_time:.2f}s")

    return targets


# ============================================================================
# Curriculum Gate  (GBFS-based, unchanged)
# ============================================================================

def should_advance_curriculum_gbfs(
    gbfs_res,
    solve_threshold=75.0,
    min_depth_fraction=0.6,
):
    """
    Advance curriculum only when enough depth bins achieve the required GBFS
    solve rate.

    Returns:
        advance          : bool
        fraction         : fraction of bins that passed
        per_depth_status : list of (depth, pct_solved, passed)
    """
    depths     = gbfs_res.get("depths",     [])
    pct_solved = gbfs_res.get("pct_solved", [])

    if not depths:
        return False, 0.0, []

    passed           = 0
    per_depth_status = []

    for d, p in zip(depths, pct_solved):
        ok = p >= solve_threshold
        if ok:
            passed += 1
        per_depth_status.append((d, p, ok))

    fraction = passed / len(depths)
    advance  = fraction >= min_depth_fraction

    print(
        f"  Curriculum gate (GBFS): {passed}/{len(depths)} depth bins "
        f">= {solve_threshold:.0f}% solved ({fraction * 100:.1f}%, "
        f"need {min_depth_fraction * 100:.0f}%) -> "
        f"{'ADVANCE' if advance else 'STAY'}"
    )

    return advance, fraction, per_depth_status


def infer_curriculum_stage(current_back_max, curriculum_schedule):
    """Infer the curriculum stage index from a back_max value on resume."""
    for idx, depth in enumerate(curriculum_schedule):
        if current_back_max <= depth:
            return idx
    return len(curriculum_schedule) - 1


# ============================================================================
# Checkpointing
# ============================================================================

def _plain_state_dict(net):
    """Return state_dict with _orig_mod. prefix stripped (added by torch.compile)."""
    sd = net.state_dict()
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    return sd


def save_checkpoint(
    current_net,
    target_net,
    optimizer,
    scaler,
    itr,
    update_num,
    current_back_max,
    curriculum_stage_idx,
    stage_update_cycle=0,
    kl_history=None,
    use_all_zeros=False,
    save_dir="saved_models",
):
    """
    Persist full training state to disk.

    Saves both network state dicts individually (for solve.py compatibility)
    and a comprehensive pickle with all optimizer / curriculum state, including:
      - kl_history    : sliding window of per-cycle training KL for stagnation
      - use_all_zeros : whether target is still in zero-initialization phase
    """
    os.makedirs(os.path.join(save_dir, "current"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "target"),  exist_ok=True)

    torch.save(
        _plain_state_dict(current_net),
        os.path.join(save_dir, "current", "model_state_dict.pt"),
    )
    torch.save(
        _plain_state_dict(target_net),
        os.path.join(save_dir, "target", "model_state_dict.pt"),
    )

    checkpoint = {
        "itr":                  itr,
        "update_num":           update_num,
        "current_back_max":     current_back_max,
        "curriculum_stage_idx": curriculum_stage_idx,
        "stage_update_cycle":   stage_update_cycle,
        # Stagnation tracking
        "kl_history":           list(kl_history) if kl_history is not None else [],
        # all_zeros mode flag
        "use_all_zeros":        use_all_zeros,
        # Network and optimizer state
        "current_net_state":    _plain_state_dict(current_net),
        "target_net_state":     _plain_state_dict(target_net),
        "optimizer_state":      optimizer.state_dict(),
        "scaler_state":         scaler.state_dict(),
    }

    with open(os.path.join(save_dir, "training_state.pkl"), "wb") as f:
        pickle.dump(checkpoint, f)


# ============================================================================
# GBFS Evaluation  (reusable helper)
# ============================================================================

def run_gbfs_evaluation(
    env,
    current_net,
    device,
    args,
    current_back_max,
    child_eval_chunk,
    scramble_chunk,
    itr,
):
    """Run GBFS, log results, and update the training-progress plot."""
    print(f"\nRunning GBFS evaluation (itr={itr}, depth={current_back_max})...")

    gbfs_res = test_gbfs_gpu(
        env=env,
        current_net=current_net,
        device=device,
        num_test=args.gbfs_num_test,
        back_max=current_back_max,
        max_solve_steps=current_back_max + 30,
        child_eval_chunk=child_eval_chunk,
        scramble_chunk=scramble_chunk,
    )

    history_path = os.path.join(args.save_dir, "gbfs_history.json")
    plot_path    = os.path.join(args.save_dir, "training_progress.png")

    append_gbfs_result(gbfs_res, itr, current_back_max, filepath=history_path)
    plot_training_progress(filepath=history_path, output=plot_path)

    return gbfs_res


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    profile = get_hw_profile(device)

    TRAIN_BATCH_SIZE = profile["TRAIN_BATCH_SIZE"]
    EVAL_BATCH_SIZE  = profile["EVAL_BATCH_SIZE"]
    CHILD_EVAL_CHUNK = profile["CHILD_EVAL_CHUNK"]
    SCRAMBLE_CHUNK   = profile["SCRAMBLE_CHUNK"]

    env = MalteseGearCubeEnv()
    env.setup_gpu(device, max_chunk_size=max(TRAIN_BATCH_SIZE, CHILD_EVAL_CHUNK))

    os.makedirs(os.path.join(args.save_dir, "current"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "target"),  exist_ok=True)
    sys.stdout = Logger(os.path.join(args.save_dir, "training_log.txt"))

    # -- Banner ---------------------------------------------------------------
    print("=" * 70)
    print("Maltese Gear Cube  --  DAVI Training (Training-KL Target Sync)")
    print("=" * 70)
    print(f"Device                : {device}")
    print(f"Curriculum            : {args.curriculum}")
    print(f"MAX_DIST / sigma      : {args.max_dist} / {args.sigma}")
    print(f"LR / LR_decay         : {args.lr} / {args.lr_decay}")
    print(f"States / update cycle : {args.states_per_update:,}")
    print(f"Max iterations        : {args.max_itrs:,}")
    print(f"Batch sizes  Train={TRAIN_BATCH_SIZE}  Eval={EVAL_BATCH_SIZE}")
    print(f"Grad clip             : {args.grad_clip}")
    print(
        f"KL sync threshold     : {args.kl_base} * "
        f"(back_max / {args.curriculum[0]}) ^ {args.kl_scale_power}"
    )
    print(
        f"Stagnation detection  : patience={args.kl_stagnation_patience} "
        f"cycles, min_delta={args.kl_stagnation_min_delta}"
    )
    print(f"Max cycles per stage  : {args.max_stage_cycles}")
    print(f"GBFS eval every       : {args.gbfs_eval_freq} update cycles")
    print(
        f"Curriculum gate       : GBFS >= {args.gbfs_solve_threshold:.0f}% on "
        f"{args.min_solve_depth_fraction * 100:.0f}% of depth bins"
    )
    print("=" * 70)

    # -- Model instantiation --------------------------------------------------
    def build_net():
        return CategoricalResNet(
            max_dist=args.max_dist,
            hidden_dim=args.hidden_dim,
            res_dim=args.res_dim,
            num_blocks=args.num_res_blocks,
        ).to(device)

    current_net = build_net()
    target_net  = build_net()

    # -----------------------------------------------------------------
    # Compilation strategy: max-autotune-no-cudagraphs
    #
    # CRITICAL: We do NOT use mode="reduce-overhead" for ANY network.
    # reduce-overhead enables CUDAGraph Trees, which require a consistent
    # CUDA memory pool liveness state between graph recording and every
    # replay.  Our training loop calls torch.cuda.empty_cache() between
    # update cycles (to free 12M-state training tensors), which resets the
    # pool and invalidates all previously recorded CUDAGraphs.  The Trees
    # system then re-records — allocating ~1 GB of new persistent static-
    # buffer memory per cycle — and keeps old recordings alive as parent
    # nodes.  This is confirmed as PyTorch issue #159669 and #128424.
    #
    # max-autotune-no-cudagraphs gives us:
    #   - Triton kernel fusion (the real speedup for Linear+BN+ReLU)
    #   - Matmul autotuning (benchmarks multiple implementations)
    #   - NO CUDAGraphs (no VRAM leak, no re-recording)
    #
    # target_net stays eager — it runs under torch.no_grad() in the
    # Bellman loop where compilation adds no measurable benefit.
    #
    # References:
    #   PyTorch issue #159669 — CUDAGraph VRAM leak
    #   PyTorch issue #128424 — reduce-overhead memory growth
    #   PyTorch issue #171672 — CUDAGraph rebuild every iteration
    # -----------------------------------------------------------------
    if hasattr(torch, "compile") and not args.no_compile and sys.platform != "win32":
        print("Compiling current_net with torch.compile (max-autotune-no-cudagraphs)...")
        print("  [NO CUDAGraphs — prevents VRAM leak from graph re-recording]")
        print("  [target_net stays eager — no compilation benefit under no_grad]")
        current_net = torch.compile(current_net, mode="max-autotune-no-cudagraphs")

    optimizer = torch.optim.Adam(current_net.parameters(), lr=args.lr)
    scaler    = make_scaler(device)

    # -- Pre-allocated VRAM buffers -------------------------------------------
    # ctg_buf: used inside compute_bellman_targets_gpu to avoid torch.empty
    # in the hot loop.  Sized for the maximum number of children per outer
    # batch: EVAL_BATCH_SIZE * num_moves.
    ctg_buf = torch.empty(
        EVAL_BATCH_SIZE * env.num_moves,
        dtype=torch.float32,
        device=device,
    )

    # -- Checkpoint resume ----------------------------------------------------
    checkpoint_path    = os.path.join(args.save_dir, "training_state.pkl")
    resume_stage_cycle = 0
    is_fresh_start     = not os.path.exists(checkpoint_path)

    if not is_fresh_start:
        print("\n=== Checkpoint found -- resuming training ===")
        with open(checkpoint_path, "rb") as f:
            state = pickle.load(f)

        if "current_net_state" in state:
            current_net.load_state_dict(state["current_net_state"])
            target_net.load_state_dict(state["target_net_state"])
        else:
            current_net.load_state_dict(
                torch.load(
                    os.path.join(args.save_dir, "current", "model_state_dict.pt"),
                    map_location=device,
                )
            )
            target_net.load_state_dict(
                torch.load(
                    os.path.join(args.save_dir, "target", "model_state_dict.pt"),
                    map_location=device,
                )
            )

        if "optimizer_state" in state:
            optimizer.load_state_dict(state["optimizer_state"])
        if "scaler_state" in state:
            scaler.load_state_dict(state["scaler_state"])

        itr                  = state.get("itr", 0)
        update_num           = state.get("update_num", 0)
        current_back_max     = state.get("current_back_max", args.curriculum[0])
        curriculum_stage_idx = state.get(
            "curriculum_stage_idx",
            infer_curriculum_stage(current_back_max, args.curriculum),
        )
        resume_stage_cycle   = state.get("stage_update_cycle", 0)

        # Restore stagnation tracking
        saved_kl_history = state.get("kl_history", [])
        kl_history = collections.deque(
            saved_kl_history[-args.kl_stagnation_patience:],
            maxlen=args.kl_stagnation_patience,
        )

        # Restore all_zeros flag
        use_all_zeros = state.get("use_all_zeros", False)

        print(
            f"Resumed at itr={itr}, update_num={update_num}, "
            f"depth={current_back_max}, stage={curriculum_stage_idx}, "
            f"stage_cycle={resume_stage_cycle}, "
            f"use_all_zeros={use_all_zeros}, "
            f"kl_history_len={len(kl_history)}"
        )

    else:
        # -----------------------------------------------------------------
        # Fresh start: target_net weights are IGNORED until first sync.
        # We use all_zeros=True for every Bellman target computation until
        # the first sync fires (matching DeepCubeA's all_zeros behavior).
        # -----------------------------------------------------------------
        print("\n=== No checkpoint found -- starting fresh ===")
        target_net.load_state_dict(_plain_state_dict(current_net))

        itr                  = 0
        update_num           = 0
        curriculum_stage_idx = 0
        current_back_max     = args.curriculum[0]
        kl_history           = collections.deque(maxlen=args.kl_stagnation_patience)
        use_all_zeros        = True

        print(
            "Target network is in ALL-ZEROS mode until first KL- or "
            "stagnation-triggered sync (DeepCubeA initialization)."
        )

    # =========================================================================
    # Outer curriculum loop
    # =========================================================================
    while itr < args.max_itrs and curriculum_stage_idx < len(args.curriculum):
        current_back_max = args.curriculum[curriculum_stage_idx]

        # KL sync threshold is fixed for the entire duration of this stage
        kl_threshold = get_avg_kl_sync_threshold(
            back_max=current_back_max,
            base_threshold=args.kl_base,
            first_stage_max=args.curriculum[0],
            scale_power=args.kl_scale_power,
        )

        print(
            f"\n{'=' * 60}\n"
            f"CURRICULUM STAGE {curriculum_stage_idx + 1}/{len(args.curriculum)} "
            f"| back_max={current_back_max} "
            f"| KL sync threshold={kl_threshold:.4f}"
            f"| all_zeros={use_all_zeros}\n"
            f"{'=' * 60}"
        )

        last_gbfs_res      = None
        start_cycle        = resume_stage_cycle
        resume_stage_cycle = 0          # only skip on first resume entry

        # =====================================================================
        # Middle update-cycle loop
        # =====================================================================
        for update_cycle in range(start_cycle, args.max_stage_cycles):
            if itr >= args.max_itrs:
                break

            print(
                f"\n--- Update Cycle {update_cycle + 1}/{args.max_stage_cycles} "
                f"(stage {curriculum_stage_idx + 1}/{len(args.curriculum)}, "
                f"back_max={current_back_max}, itr={itr}, "
                f"all_zeros={use_all_zeros}, "
                f"kl_history_len={len(kl_history)}) ---"
            )

            # -----------------------------------------------------------------
            # Step 0 : VRAM cleanup from previous cycle
            # -----------------------------------------------------------------
            if device.type == "cuda":
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            # -----------------------------------------------------------------
            # Step 1 : Generate training states
            # -----------------------------------------------------------------
            print(f"Generating {args.states_per_update:,} training states on GPU...")
            gen_start     = time.time()
            states_tensor = env.generate_scrambled_states_gpu(
                args.states_per_update,
                current_back_max,
                chunk_size=SCRAMBLE_CHUNK,
            )
            print(f"State generation: {time.time() - gen_start:.2f}s")

            # -----------------------------------------------------------------
            # Step 2 : Bellman targets (all_zeros or frozen target network)
            # -----------------------------------------------------------------
            targets_tensor = compute_bellman_targets_gpu(
                env,
                target_net,
                states_tensor,
                EVAL_BATCH_SIZE,
                device,
                child_eval_chunk=CHILD_EVAL_CHUNK,
                ctg_buf=ctg_buf,
                all_zeros=use_all_zeros,
                verbose=True,
            )
            print(
                f"Targets  Mean={targets_tensor.mean().item():.2f}  "
                f"Min={targets_tensor.min().item():.2f}  "
                f"Max={targets_tensor.max().item():.2f}"
            )

            # -----------------------------------------------------------------
            # Step 3 : Inner training loop  (epochs + early stopping)
            # -----------------------------------------------------------------
            num_states       = states_tensor.size(0)
            best_epoch_loss  = float("inf")
            patience_counter = 0
            final_epoch_loss = None

            for epoch in range(args.max_inner_epochs):
                if itr >= args.max_itrs:
                    break

                current_net.train()
                epoch_total_loss = 0.0
                batch_count      = 0
                train_start      = time.time()

                indices = torch.randperm(num_states, device=device)

                for b in range(0, num_states, TRAIN_BATCH_SIZE):
                    batch_idx   = indices[b : b + TRAIN_BATCH_SIZE]
                    batch_x_raw = states_tensor[batch_idx]
                    batch_y     = targets_tensor[batch_idx]

                    batch_x      = env.states_to_nnet_input_static(batch_x_raw)
                    target_probs = create_soft_targets(
                        batch_y, args.max_dist, device, sigma=args.sigma
                    )

                    # Per-step exponential LR decay
                    current_lr = args.lr * (args.lr_decay ** itr)
                    for pg in optimizer.param_groups:
                        pg["lr"] = current_lr

                    optimizer.zero_grad(set_to_none=True)

                    with autocast_ctx(device):
                        logits    = current_net(batch_x)
                        log_probs = F.log_softmax(logits, dim=-1)
                        loss      = F.kl_div(
                            log_probs, target_probs, reduction="batchmean"
                        )

                    scaler.scale(loss).backward()

                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            current_net.parameters(), args.grad_clip
                        )

                    scaler.step(optimizer)
                    scaler.update()

                    loss_val          = loss.item()
                    epoch_total_loss += loss_val
                    batch_count      += 1
                    itr              += 1

                    if itr % 100 == 0:
                        pred_ctg = current_net.get_ctg(logits).mean().item()
                        print(
                            f"  Itr: {itr}  lr: {current_lr:.2E}  "
                            f"KL: {loss_val:.4f}  "
                            f"targ: {batch_y.mean().item():.2f}  "
                            f"pred: {pred_ctg:.2f}  "
                            f"time: {time.time() - train_start:.2f}s"
                        )
                        train_start = time.time()

                    if itr >= args.max_itrs:
                        break

                avg_epoch_loss = epoch_total_loss / max(1, batch_count)
                final_epoch_loss = avg_epoch_loss

                print(
                    f"  Epoch {epoch + 1}/{args.max_inner_epochs} | "
                    f"Train KL: {avg_epoch_loss:.4f} | itr: {itr}"
                )

                # Early stopping within the inner loop
                if avg_epoch_loss < best_epoch_loss * (1.0 - args.min_improvement):
                    best_epoch_loss  = avg_epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= args.early_stop_patience:
                        print(
                            f"  Early stop: no improvement for "
                            f"{args.early_stop_patience} epochs"
                        )
                        break

            # -----------------------------------------------------------------
            # Step 4 : Free training tensors before sync / GBFS
            # -----------------------------------------------------------------
            del states_tensor, targets_tensor, indices
            if device.type == "cuda":
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            # Guard: if inner loop produced no loss (e.g., max_itrs hit early)
            if final_epoch_loss is None:
                break

            # -----------------------------------------------------------------
            # Step 5 : Target-network sync decision
            #
            # Two triggers:
            #   1. KL threshold: training KL < kl_threshold
            #   2. Stagnation:   relative improvement over sliding window
            #                    < kl_stagnation_min_delta
            # -----------------------------------------------------------------
            kl_history.append(final_epoch_loss)

            kl_triggered = final_epoch_loss < kl_threshold

            # Stagnation check: need a full window of data
            stagnation_triggered = False
            if len(kl_history) >= kl_history.maxlen:
                oldest = kl_history[0]
                newest = kl_history[-1]
                if oldest > 0:
                    relative_improvement = (oldest - newest) / abs(oldest)
                else:
                    relative_improvement = 0.0
                stagnation_triggered = relative_improvement < args.kl_stagnation_min_delta
            else:
                relative_improvement = float("nan")

            sync_reason = None
            if kl_triggered:
                sync_reason = "KL"
            elif stagnation_triggered:
                sync_reason = "STAGNATION"

            if sync_reason:
                target_net.load_state_dict(_plain_state_dict(current_net))
                use_all_zeros = False      # permanently disable
                kl_history.clear()         # reset window for next phase

            print(
                f"Sync decision | "
                f"train_kl={final_epoch_loss:.4f} | threshold={kl_threshold:.4f} | "
                f"kl_triggered={kl_triggered} | "
                f"stagnation_triggered={stagnation_triggered} "
                f"(rel_improv={relative_improvement:.4f} over {len(kl_history)}/{args.kl_stagnation_patience} cycles) | "
                f"result={'SYNCED [' + sync_reason + ']' if sync_reason else 'NO SYNC'}"
            )

            if sync_reason:
                print(
                    f">>> Target network synced [{sync_reason}] "
                    f"| update_cycle={update_cycle + 1} "
                    f"| train_kl={final_epoch_loss:.4f} "
                    f"| threshold={kl_threshold:.4f} "
                    f"| all_zeros now={use_all_zeros}"
                )

            update_num += 1

            # -----------------------------------------------------------------
            # Step 6 : GBFS evaluation  (runs on a fixed schedule)
            # -----------------------------------------------------------------
            run_gbfs = (update_num % args.gbfs_eval_freq == 0)

            if run_gbfs:
                last_gbfs_res = run_gbfs_evaluation(
                    env=env,
                    current_net=current_net,
                    device=device,
                    args=args,
                    current_back_max=current_back_max,
                    child_eval_chunk=CHILD_EVAL_CHUNK,
                    scramble_chunk=SCRAMBLE_CHUNK,
                    itr=itr,
                )

            # -----------------------------------------------------------------
            # Step 7 : Checkpoint every cycle
            # -----------------------------------------------------------------
            save_checkpoint(
                current_net=current_net,
                target_net=target_net,
                optimizer=optimizer,
                scaler=scaler,
                itr=itr,
                update_num=update_num,
                current_back_max=current_back_max,
                curriculum_stage_idx=curriculum_stage_idx,
                stage_update_cycle=update_cycle + 1,
                kl_history=kl_history,
                use_all_zeros=use_all_zeros,
                save_dir=args.save_dir,
            )

            # -----------------------------------------------------------------
            # Step 8 : Curriculum advancement check  (only after GBFS run)
            # -----------------------------------------------------------------
            if run_gbfs and last_gbfs_res is not None:
                advance, frac_passed, _ = should_advance_curriculum_gbfs(
                    last_gbfs_res,
                    solve_threshold=args.gbfs_solve_threshold,
                    min_depth_fraction=args.min_solve_depth_fraction,
                )

                if advance:
                    curriculum_stage_idx += 1

                    if curriculum_stage_idx < len(args.curriculum):
                        next_depth     = args.curriculum[curriculum_stage_idx]
                        next_threshold = get_avg_kl_sync_threshold(
                            back_max=next_depth,
                            base_threshold=args.kl_base,
                            first_stage_max=args.curriculum[0],
                            scale_power=args.kl_scale_power,
                        )
                        print(
                            f"\n=== CURRICULUM ADVANCE -> back_max={next_depth} "
                            f"(stage {curriculum_stage_idx + 1}/"
                            f"{len(args.curriculum)}) ==="
                        )
                        print(f"    New KL sync threshold = {next_threshold:.4f}")
                    else:
                        print("\n=== All curriculum stages completed! ===")

                    # Reset stagnation window for the incoming stage
                    kl_history.clear()

                    save_checkpoint(
                        current_net=current_net,
                        target_net=target_net,
                        optimizer=optimizer,
                        scaler=scaler,
                        itr=itr,
                        update_num=update_num,
                        current_back_max=args.curriculum[
                            min(curriculum_stage_idx, len(args.curriculum) - 1)
                        ],
                        curriculum_stage_idx=curriculum_stage_idx,
                        stage_update_cycle=0,
                        kl_history=kl_history,
                        use_all_zeros=use_all_zeros,
                        save_dir=args.save_dir,
                    )
                    break   # exit middle loop; outer loop starts new stage

                else:
                    print(
                        f"  GBFS gate not passed "
                        f"({frac_passed * 100:.1f}% of depth bins >= "
                        f"{args.gbfs_solve_threshold:.0f}% solved, "
                        f"need {args.min_solve_depth_fraction * 100:.0f}%). "
                        f"Continuing at back_max={current_back_max}."
                    )

        else:
            # -----------------------------------------------------------------
            # max_stage_cycles exhausted without GBFS advancement.
            # Save checkpoint with stage_update_cycle=0 so the outer loop
            # restarts the same stage from cycle 0.
            # -----------------------------------------------------------------
            print(
                f"\n[WARNING] max_stage_cycles ({args.max_stage_cycles}) exhausted "
                f"at back_max={current_back_max} without GBFS advancement.\n"
                f"Saving checkpoint and restarting stage from cycle 0.\n"
                f"Consider raising --max_stage_cycles or reviewing GBFS solve rate."
            )
            save_checkpoint(
                current_net=current_net,
                target_net=target_net,
                optimizer=optimizer,
                scaler=scaler,
                itr=itr,
                update_num=update_num,
                current_back_max=current_back_max,
                curriculum_stage_idx=curriculum_stage_idx,
                stage_update_cycle=0,
                kl_history=kl_history,
                use_all_zeros=use_all_zeros,
                save_dir=args.save_dir,
            )

    print(f"\nTraining finished. Total gradient steps: {itr}")


if __name__ == "__main__":
    main()
