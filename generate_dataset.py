"""
Dataset Generator for Maltese Gear Cube

Generates scrambled cube states for testing/benchmarking a trained model.
States are saved as a .pt file containing a dict with:
  - "states": (N, 144) uint8 tensor of scrambled states
  - "depths": (N,) long tensor of scramble depths
  - "metadata": dict with generation parameters

Unlike the training scrambler, this generator does NOT perform inverse-move
avoidance. This means some effective scramble depths may be shorter than the
nominal depth due to self-cancelling pairs (e.g., L followed by L').
This is intentional for testing — it produces a more realistic distribution
of "real-world" scrambles.

Usage:
  python generate_dataset.py --num_states 10000 --min_depth 1 --max_depth 100 \\
                              --output test_dataset.pt
"""

import argparse
import time

import numpy as np
import torch

from environment import MalteseGearCubeEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate scrambled Maltese Gear Cube states for testing."
    )
    parser.add_argument("--num_states", type=int, required=True,
                        help="Number of scrambled states to generate")
    parser.add_argument("--min_depth", type=int, default=1,
                        help="Minimum scramble depth (default: 1)")
    parser.add_argument("--max_depth", type=int, required=True,
                        help="Maximum scramble depth")
    parser.add_argument("--output", type=str, default="test_dataset.pt",
                        help="Output file path (default: test_dataset.pt)")
    parser.add_argument("--chunk_size", type=int, default=500_000,
                        help="GPU chunk size for generation (default: 500000)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cuda', 'cpu', or 'auto' (default: auto)")
    return parser.parse_args()


def generate_scrambled_states_no_avoidance(
    env, num_states, min_depth, max_depth, chunk_size=500_000
):
    """
    Generate scrambled states WITHOUT inverse-move avoidance.

    Each state is scrambled from the goal state by a random number of moves
    sampled uniformly from [min_depth, max_depth]. Moves are chosen uniformly
    at random with no filtering of inverse pairs.

    Args:
        env: MalteseGearCubeEnv with GPU setup complete
        num_states: total states to generate
        min_depth: minimum scramble depth (inclusive)
        max_depth: maximum scramble depth (inclusive)
        chunk_size: GPU processing chunk size

    Returns:
        states: (num_states, 144) uint8 tensor on GPU
        depths: (num_states,) long tensor on GPU
    """
    device = env.device
    all_states = torch.empty((num_states, 144), dtype=torch.uint8, device=device)
    all_depths = torch.empty(num_states, dtype=torch.long, device=device)

    for offset in range(0, num_states, chunk_size):
        size = min(chunk_size, num_states - offset)
        states = env.goal_colors_tensor.unsqueeze(0).expand(size, 144).clone()

        # Sample depths uniformly from [min_depth, max_depth]
        target_lengths = torch.randint(
            min_depth, max_depth + 1, (size,), device=device, dtype=torch.long
        )

        current_lengths = torch.zeros(size, device=device, dtype=torch.long)
        max_len = target_lengths.max().item() if size > 0 else 0

        for _ in range(max_len):
            active_mask = current_lengths < target_lengths
            if not active_mask.any():
                break

            # Pure random actions — NO inverse-move avoidance
            actions = torch.randint(0, env.num_moves, (size,), device=device)

            # Apply moves
            action_idx = env.T_idx_old[actions]
            next_states = torch.gather(states, 1, action_idx)

            gears = next_states[:, 120:].to(torch.int16)
            adds = env.T_adds[actions, 120:]
            next_states[:, 120:] = torch.remainder(gears + adds, 4).to(torch.uint8)

            states = torch.where(active_mask.unsqueeze(1), next_states, states)
            current_lengths += active_mask.long()

        all_states[offset : offset + size] = states
        all_depths[offset : offset + size] = target_lengths

    return all_states, all_depths


def main():
    args = parse_args()

    # Validate
    assert args.min_depth >= 0, "min_depth must be >= 0"
    assert args.max_depth >= args.min_depth, "max_depth must be >= min_depth"
    assert args.num_states > 0, "num_states must be > 0"

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Generating {args.num_states:,} states with depths [{args.min_depth}, {args.max_depth}]")

    # Setup environment
    env = MalteseGearCubeEnv()
    env.setup_gpu(device, max_chunk_size=min(args.chunk_size, args.num_states))

    # Generate
    start_time = time.time()
    states, depths = generate_scrambled_states_no_avoidance(
        env, args.num_states, args.min_depth, args.max_depth,
        chunk_size=args.chunk_size,
    )
    elapsed = time.time() - start_time

    # Save (move to CPU for portable .pt file)
    dataset = {
        "states": states.cpu(),
        "depths": depths.cpu(),
        "metadata": {
            "num_states": args.num_states,
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            "inverse_move_avoidance": False,
        },
    }
    torch.save(dataset, args.output)

    # Summary
    print(f"\nGeneration complete in {elapsed:.2f}s")
    print(f"States shape: {states.shape}")
    print(f"Depth distribution: min={depths.min().item()}, "
          f"max={depths.max().item()}, mean={depths.float().mean().item():.1f}")

    # Quick sanity check: how many are actually solved?
    num_solved = env.is_solved_gpu(states).sum().item()
    print(f"Solved states in dataset: {num_solved}/{args.num_states} "
          f"({100 * num_solved / args.num_states:.2f}%)")

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
