"""
Maltese Gear Cube Solver — Batch Weighted A* Search (BWAS)

Takes a cube state and uses a trained CategoricalResNet + BWAS to find
a solution (sequence of moves to reach the goal state).

State input methods:
  1. --state_file: Load a 144-element uint8 numpy array from .npy file
  2. --dataset: Load a .pt dataset (from generate_dataset.py), solve index --idx
  3. --scramble_depth: Generate a random scramble of given depth and solve it
  4. --state_csv: Comma-separated 144 integers on the command line

Usage examples:
  # Solve a random scramble of depth 50
  python solve.py --model saved_models/current/model_state_dict.pt \\
                  --scramble_depth 50

  # Solve a specific state from a test dataset
  python solve.py --model saved_models/current/model_state_dict.pt \\
                  --dataset test_dataset.pt --idx 42

  # Solve a state provided as .npy file
  python solve.py --model saved_models/current/model_state_dict.pt \\
                  --state_file my_state.npy

Reference:
  Agostinelli et al. (2019), Algorithm 2: Batch Weighted A* Search.
"""

import argparse
import sys
import time

import numpy as np
import torch

from environment import MalteseGearCubeEnv
from model import CategoricalResNet
from search import batch_weighted_astar_search


def parse_args():
    parser = argparse.ArgumentParser(
        description="Solve a Maltese Gear Cube state using BWAS with a trained model."
    )

    # --- Model ---
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model state_dict (.pt file)")
    parser.add_argument("--max_dist", type=int, default=505,
                        help="Model max_dist (must match training) (default: 505)")
    parser.add_argument("--hidden_dim", type=int, default=7000,
                        help="Model hidden_dim (must match training) (default: 7000)")
    parser.add_argument("--res_dim", type=int, default=2000,
                        help="Model res_dim (must match training) (default: 2000)")
    parser.add_argument("--num_res_blocks", type=int, default=6,
                        help="Model num_res_blocks (must match training) (default: 6)")

    # --- State input (mutually exclusive) ---
    state_group = parser.add_mutually_exclusive_group(required=True)
    state_group.add_argument("--state_file", type=str,
                             help="Path to .npy file containing a 144-element uint8 array")
    state_group.add_argument("--dataset", type=str,
                             help="Path to .pt dataset file (from generate_dataset.py)")
    state_group.add_argument("--scramble_depth", type=int,
                             help="Generate a random scramble of this depth and solve it")
    state_group.add_argument("--state_csv", type=str,
                             help="Comma-separated 144 integers (0-5 for colors, 0-3 for gears)")

    # --- Dataset index ---
    parser.add_argument("--idx", type=int, default=0,
                        help="Index into dataset when using --dataset (default: 0)")

    # --- BWAS parameters ---
    parser.add_argument("--weight", type=float, default=0.6,
                        help="Heuristic weight λ. Lower → more optimal but slower. "
                             "Range [0, 1]. (default: 0.6)")
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Nodes expanded per GPU batch (default: 1000)")
    parser.add_argument("--max_expansions", type=int, default=100_000,
                        help="Maximum total node expansions (default: 100000)")

    # --- Device ---
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cuda', 'cpu', or 'auto' (default: auto)")

    return parser.parse_args()


def load_state(args, env):
    """Load or generate the initial state based on command-line arguments."""
    if args.state_file:
        state = np.load(args.state_file).astype(np.uint8)
        assert state.shape == (144,), f"Expected shape (144,), got {state.shape}"
        print(f"Loaded state from {args.state_file}")
        return state

    elif args.dataset:
        dataset = torch.load(args.dataset, map_location="cpu")
        states = dataset["states"]
        depths = dataset["depths"]
        assert args.idx < len(states), (
            f"Index {args.idx} out of range (dataset has {len(states)} states)"
        )
        state = states[args.idx].numpy().astype(np.uint8)
        depth = depths[args.idx].item()
        print(f"Loaded state index {args.idx} from {args.dataset} (scramble depth: {depth})")
        return state

    elif args.scramble_depth is not None:
        print(f"Generating random scramble of depth {args.scramble_depth}...")
        states = env.generate_scrambled_states_gpu(
            1, args.scramble_depth, exact_moves=args.scramble_depth
        )
        state = states[0].cpu().numpy().astype(np.uint8)
        return state

    elif args.state_csv:
        values = [int(x.strip()) for x in args.state_csv.split(",")]
        assert len(values) == 144, f"Expected 144 values, got {len(values)}"
        state = np.array(values, dtype=np.uint8)
        print("Loaded state from command-line CSV")
        return state

    else:
        print("Error: No state input provided.")
        sys.exit(1)


def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")

    # Environment
    env = MalteseGearCubeEnv()
    env.setup_gpu(device, max_chunk_size=50_000)

    # Load model
    print(f"Loading model from {args.model}...")
    net = CategoricalResNet(
        max_dist=args.max_dist,
        hidden_dim=args.hidden_dim,
        res_dim=args.res_dim,
        num_blocks=args.num_res_blocks,
    ).to(device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    net.eval()
    print("Model loaded successfully.")

    # Load state
    initial_state = load_state(args, env)

    # Display initial state info
    initial_tensor = torch.from_numpy(initial_state[None, :]).to(device)
    is_solved = env.is_solved_gpu(initial_tensor)[0].item()
    print(f"\nInitial state solved: {is_solved}")

    if is_solved:
        print("State is already solved! No moves needed.")
        return

    # Run BWAS
    print(f"\nRunning Batch Weighted A* Search...")
    print(f"  Weight (λ): {args.weight}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max expansions: {args.max_expansions:,}")
    print()

    start_time = time.time()
    solution = batch_weighted_astar_search(
        env=env,
        net=net,
        device=device,
        initial_state_np=initial_state,
        weight=args.weight,
        batch_size=args.batch_size,
        max_expansions=args.max_expansions,
    )
    elapsed = time.time() - start_time

    # Report results
    print(f"\n{'=' * 50}")
    if solution is not None:
        print(f"SOLUTION FOUND in {elapsed:.2f}s")
        print(f"Solution length: {len(solution)} moves")
        print(f"Moves: {' '.join(solution)}")

        # Verify solution
        print("\nVerifying solution...", end=" ")
        state = initial_state.copy()
        state_tensor = torch.from_numpy(state[None, :]).to(device)
        for move_str in solution:
            move_idx = env.moves.index(move_str)
            idx = env.T_idx_old[move_idx].unsqueeze(0)
            state_tensor = torch.gather(state_tensor, 1, idx)
            gears = state_tensor[:, 120:].to(torch.int16)
            adds = env.T_adds[move_idx, 120:].unsqueeze(0)
            state_tensor[:, 120:] = torch.remainder(gears + adds, 4).to(torch.uint8)

        verified = env.is_solved_gpu(state_tensor)[0].item()
        print("VERIFIED ✓" if verified else "FAILED ✗")
    else:
        print(f"NO SOLUTION FOUND in {elapsed:.2f}s")
        print("Try increasing --max_expansions or adjusting --weight")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
