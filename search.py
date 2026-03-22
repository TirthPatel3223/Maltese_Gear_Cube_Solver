"""
Search algorithms for the Maltese Gear Cube.

Contains:
  - test_gbfs_gpu: Greedy Best-First Search for training evaluation
  - batch_weighted_astar_search: Batch Weighted A* (BWAS) for solving

Reference:
  Agostinelli et al. (2019), "Solving the Rubik's cube with deep reinforcement
  learning and search," Nature Machine Intelligence, Algorithm 2 (BWAS).
"""

import heapq
import time

import numpy as np
import torch

from utils import autocast_ctx


# ============================================================================
# GBFS — Greedy Best-First Search (Training Evaluation)
# ============================================================================

def test_gbfs_gpu(env, current_net, device, num_test, back_max,
                  max_solve_steps, child_eval_chunk, scramble_chunk):
    """
    Evaluate the current network using Greedy Best-First Search.

    Generates states at linearly-spaced scramble depths from 0 to back_max,
    then attempts to solve each by greedily selecting the child with the
    lowest predicted cost-to-go at each step.

    Also tracks overestimation: for each depth bin, what fraction of states
    have CTG estimate > actual scramble depth. This is a diagnostic for
    how badly the network overestimates cost-to-go at each depth.

    Returns dict with per-depth:
      - pct_solved: GBFS solve percentage
      - avg_ctg: mean predicted CTG
      - pct_overestimated: fraction of states where CTG > scramble depth
    """
    print(f"Solving {num_test} states with GBFS with {max_solve_steps} steps")
    start_time = time.time()

    back_steps_list = np.unique(np.linspace(0, back_max, 500, dtype=int))
    states_per_bin = max(1, num_test // len(back_steps_list))
    current_net.eval()

    results = {
        "depths": [],
        "pct_solved": [],
        "avg_ctg": [],
        "pct_overestimated": [],
    }

    for back_steps in back_steps_list:
        states_tensor = env.generate_scrambled_states_gpu(
            states_per_bin, back_max, exact_moves=back_steps, chunk_size=scramble_chunk
        )

        with torch.no_grad():
            inputs = env.states_to_nnet_input(states_tensor)
            with autocast_ctx(device):
                logits = current_net(inputs)
            initial_ctg = current_net.get_ctg(logits)

            solved_mask = env.is_solved_gpu(states_tensor)
            initial_ctg = torch.where(solved_mask, torch.zeros_like(initial_ctg), initial_ctg)

            ctg_mean = initial_ctg.mean().item()
            ctg_std = initial_ctg.std().item() if initial_ctg.numel() > 1 else 0.0
            ctg_min = initial_ctg.min().item()
            ctg_max = initial_ctg.max().item()

            # --- Overestimation tracking ---
            # A state is overestimated if CTG prediction > actual scramble depth.
            # Depth 0 states are solved (CTG=0), so never overestimated.
            if back_steps > 0:
                overest_mask = initial_ctg > float(back_steps)
                pct_overest = overest_mask.float().mean().item() * 100.0
            else:
                pct_overest = 0.0

            solve_steps = torch.zeros(states_per_bin, dtype=torch.long, device=device)
            active_mask = ~solved_mask

            steps = 0
            while active_mask.any() and steps < max_solve_steps:
                active_states = states_tensor[active_mask]
                children = env.expand_gpu(active_states)
                B = children.size(0)
                children_flat = children.view(-1, 144)

                # Evaluate children in chunks for memory efficiency
                costs_chunks = []
                for j in range(0, children_flat.size(0), child_eval_chunk):
                    chunk = children_flat[j : j + child_eval_chunk]
                    chunk_inputs = env.states_to_nnet_input(chunk)
                    with autocast_ctx(device):
                        chunk_logits = current_net(chunk_inputs)
                    costs_chunks.append(current_net.get_ctg(chunk_logits))

                costs = torch.cat(costs_chunks, dim=0).view(B, env.num_moves)
                best_moves = torch.argmin(costs, dim=1)

                # Apply best moves
                idx_expanded = env.T_idx_old[best_moves]
                next_states = torch.gather(active_states, 1, idx_expanded)
                gears = next_states[:, 120:].to(torch.int16)
                adds = env.T_adds[best_moves][:, 120:]
                next_states[:, 120:] = torch.remainder(gears + adds, 4).to(torch.uint8)

                states_tensor[active_mask] = next_states
                new_solved = env.is_solved_gpu(states_tensor)
                just_solved = new_solved & active_mask
                solve_steps[just_solved] = steps + 1
                active_mask = ~new_solved
                steps += 1

            final_solved = env.is_solved_gpu(states_tensor)
            pct_solved = final_solved.float().mean().item() * 100.0
            avg_solve_steps = (
                solve_steps[final_solved].float().mean().item()
                if final_solved.any()
                else 0.0
            )

            print(
                f"Back Steps: {back_steps:3d}, %Solved: {pct_solved:6.2f}, "
                f"avgSolveSteps: {avg_solve_steps:5.2f}, "
                f"CTG Mean(Std/Min/Max): {ctg_mean:.2f}"
                f"({ctg_std:.2f}/{ctg_min:.2f}/{ctg_max:.2f}), "
                f"Overest: {pct_overest:5.1f}%"
            )

            results["depths"].append(int(back_steps))
            results["pct_solved"].append(pct_solved)
            results["avg_ctg"].append(ctg_mean)
            results["pct_overestimated"].append(pct_overest)

    # --- Summary ---
    total_overest = np.mean(results["pct_overestimated"]) if results["pct_overestimated"] else 0.0
    total_solved = np.mean(results["pct_solved"]) if results["pct_solved"] else 0.0
    print(
        f"Summary — Mean Solve: {total_solved:.1f}%, "
        f"Mean Overestimation: {total_overest:.1f}%"
    )
    print(f"Test time: {time.time() - start_time:.2f}\n")
    return results


# ============================================================================
# BWAS — Batch Weighted A* Search (Solving)
# ============================================================================

def batch_weighted_astar_search(
    env, net, device, initial_state_np,
    weight=1.0, batch_size=1000, max_expansions=50000
):
    """
    Batch Weighted A* Search as defined in DeepCubeA (Agostinelli et al., 2019).

    Expands batch_size nodes at a time to maximize GPU throughput.
    f(x) = g(x) + weight * h(x), where h(x) is the neural network CTG estimate.

    Args:
        env: MalteseGearCubeEnv with GPU setup complete
        net: trained CategoricalResNet
        device: torch device
        initial_state_np: (144,) uint8 numpy array of the initial state
        weight: heuristic weight (λ in paper). Higher → faster but less optimal.
        batch_size: nodes expanded per GPU batch
        max_expansions: total node expansion budget

    Returns:
        List of move strings (solution path), or None if no solution found.
    """
    net.eval()

    open_set = []           # Min-heap: (f_score, counter, g_score, state_bytes, path)
    closed_set = {}         # state_bytes -> best g_score
    counter = 0             # Tie-breaker for heap stability

    # Check if already solved
    initial_tensor = torch.from_numpy(initial_state_np[None, :]).to(device)
    if env.is_solved_gpu(initial_tensor)[0].item():
        print("  [BWAS] Initial state is already solved!")
        return []

    # Evaluate root node
    with torch.no_grad():
        inputs = env.states_to_nnet_input(initial_tensor)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = net(inputs)
            h_initial = net.get_ctg(logits).item()

    print(f"  [BWAS Root] Initial heuristic (CTG): {h_initial:.4f}")

    g = 0
    f = g + weight * h_initial
    state_bytes = initial_state_np.tobytes()

    heapq.heappush(open_set, (f, counter, g, state_bytes, []))
    closed_set[state_bytes] = g
    counter += 1

    total_expanded = 0
    batch_count = 0

    while open_set and total_expanded < max_expansions:
        batch_count += 1
        current_batch_size = min(batch_size, len(open_set))

        # Pop a batch of nodes from the open set
        batch_states = []
        batch_g = []
        batch_paths = []

        for _ in range(current_batch_size):
            _, _, g_score, s_bytes, path = heapq.heappop(open_set)

            # Skip if we already found a better path
            if closed_set.get(s_bytes, float("inf")) < g_score:
                continue

            state_array = np.frombuffer(s_bytes, dtype=np.uint8).copy()
            batch_states.append(state_array)
            batch_g.append(g_score)
            batch_paths.append(path)

        if not batch_states:
            continue

        B = len(batch_states)
        total_expanded += B

        # Expand all nodes on GPU
        batch_states_np = np.stack(batch_states)
        batch_states_gpu = torch.from_numpy(batch_states_np).to(device)
        children_gpu = env.expand_gpu(batch_states_gpu)    # (B, num_moves, 144)
        children_flat = children_gpu.view(-1, 144)          # (B * num_moves, 144)

        # Check for goal among children
        solved_mask_flat = env.is_solved_gpu(children_flat)
        if solved_mask_flat.any():
            solved_idx = torch.where(solved_mask_flat)[0][0].item()
            parent_idx = solved_idx // env.num_moves
            move_idx = solved_idx % env.num_moves
            solution = batch_paths[parent_idx] + [env.moves[move_idx]]
            print(
                f"  [BWAS] Solution found! Length: {len(solution)}, "
                f"Nodes expanded: {total_expanded}"
            )
            return solution

        # Evaluate all children with the neural network
        with torch.no_grad():
            eval_chunk = 40_000
            h_vals_list = []
            for i in range(0, len(children_flat), eval_chunk):
                chunk = children_flat[i : i + eval_chunk]
                chunk_inputs = env.states_to_nnet_input(chunk)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = net(chunk_inputs)
                    h_vals_list.append(net.get_ctg(logits))

            h_vals_flat = torch.cat(h_vals_list, dim=0).cpu().numpy()
            h_vals = h_vals_flat.reshape(B, env.num_moves)

        # Convert children to CPU for Python dict management
        children_np = children_gpu.cpu().numpy()

        # Push valid children onto the open set
        for b in range(B):
            g_child = batch_g[b] + 1
            for a in range(env.num_moves):
                child_state = children_np[b, a]
                child_bytes = child_state.tobytes()

                if child_bytes not in closed_set or g_child < closed_set[child_bytes]:
                    closed_set[child_bytes] = g_child
                    h_child = h_vals[b, a]
                    f_child = g_child + weight * h_child
                    new_path = batch_paths[b] + [env.moves[a]]
                    heapq.heappush(
                        open_set, (f_child, counter, g_child, child_bytes, new_path)
                    )
                    counter += 1

    print(f"  [BWAS] No solution found. Expanded {total_expanded} nodes.")
    return None
