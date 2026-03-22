"""
Maltese Gear Cube Environment — GPU-Accelerated

State representation: 144-dimensional uint8 vector
  - Positions 0–95:   Face sticker colors (96 stickers across 6 faces, each 0–5)
  - Positions 96–119:  Gear position permutation (24 gears)
  - Positions 120–143: Gear rotation values (24 gears, each 0–3, mod 4 arithmetic)

Network input: 816-dimensional one-hot
  - 120 color positions × 6 classes = 720
  - 24 gear positions × 4 classes = 96

Moves: U, U', L, L', F, F'  (indices 0–5, inverse pairs at i^1)

Reference:
  Agostinelli et al. (2019), "Solving the Rubik's cube with deep reinforcement
  learning and search," Nature Machine Intelligence.
"""

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================================
# MOVE DATA — Dictionary-based (replaces elif chains)
#
# Each forward move (U, L, F) is defined by three data structures:
#   1. FACE_PERMS:  permutation of positions 0–95 (sticker colors)
#   2. GEAR_POS_PERMS: permutation of positions 96–119 (gear positions)
#   3. GEAR_INCREMENTS: {+1 positions, +2 positions, +3 positions} for mod-4 gear values
#
# Inverse moves (U', L', F') are derived automatically via np.argsort.
# ============================================================================

FACE_PERMS = {
    "U": [0, 1, 2, 3, 73, 72, 4, 5, 69, 68, 8, 9, 67, 66, 65, 64,
          95, 94, 93, 92, 22, 23, 91, 90, 26, 27, 86, 87, 28, 29, 30, 31,
          44, 40, 36, 32, 45, 41, 37, 33, 46, 42, 38, 34, 47, 43, 39, 35,
          48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
          16, 17, 18, 19, 70, 71, 20, 21, 74, 75, 24, 25, 76, 77, 78, 79,
          80, 81, 82, 83, 6, 7, 84, 85, 10, 11, 88, 89, 12, 13, 14, 15],
    "L": [48, 57, 58, 3, 52, 61, 62, 7, 56, 1, 2, 11, 60, 5, 6, 15,
          32, 41, 42, 19, 36, 45, 46, 23, 40, 17, 18, 27, 44, 21, 22, 31,
          0, 9, 10, 35, 4, 13, 14, 39, 8, 33, 34, 43, 12, 37, 38, 47,
          16, 25, 26, 51, 20, 29, 30, 55, 24, 49, 50, 59, 28, 53, 54, 63,
          76, 72, 68, 64, 77, 73, 69, 65, 78, 74, 70, 66, 79, 75, 71, 67,
          80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
    "F": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
          28, 24, 20, 16, 29, 25, 21, 17, 30, 26, 22, 18, 31, 27, 23, 19,
          32, 33, 34, 35, 69, 65, 36, 37, 70, 66, 40, 41, 79, 75, 71, 67,
          83, 87, 91, 95, 54, 55, 82, 86, 58, 59, 81, 85, 60, 61, 62, 63,
          64, 73, 74, 48, 68, 77, 78, 49, 72, 56, 52, 50, 76, 57, 53, 51,
          80, 89, 90, 47, 84, 93, 94, 46, 88, 39, 43, 45, 92, 38, 42, 44],
}

GEAR_POS_PERMS = {
    "U":  [3, 0, 1, 2, 7, 4, 5, 6, 15, 8, 9, 10, 11, 12, 13, 14,
           16, 17, 18, 19, 20, 21, 22, 23],
    "U'": [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 12, 13, 14, 15, 8,
           16, 17, 18, 19, 20, 21, 22, 23],
    "L":  [6, 1, 0, 12, 2, 11, 18, 7, 8, 3, 5, 17, 23, 13, 14, 15,
           4, 10, 22, 19, 16, 21, 20, 9],
    "L'": [2, 1, 4, 9, 16, 10, 0, 7, 8, 23, 17, 5, 3, 13, 14, 15,
           20, 11, 6, 19, 22, 21, 18, 12],
    "F":  [0, 3, 10, 5, 9, 17, 6, 1, 4, 16, 20, 11, 12, 13, 14, 2,
           8, 23, 18, 7, 15, 19, 22, 21],
    "F'": [0, 7, 15, 1, 8, 3, 6, 19, 16, 4, 2, 11, 12, 13, 14, 20,
           9, 5, 18, 21, 10, 23, 22, 17],
}

# Gear rotation increments: g1 = +1 mod 4, g2 = +2 mod 4, g3 = +3 mod 4
# Applied BEFORE the gear position permutation.
GEAR_INCREMENTS = {
    "U":  {"g1": [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15], "g2": [], "g3": []},
    "U'": {"g1": [], "g2": [], "g3": [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]},
    "L":  {"g1": [5, 11, 17, 10, 0, 2, 4, 16, 20, 18], "g2": [23, 12], "g3": [22, 6]},
    "L'": {"g1": [0, 18], "g2": [3, 12], "g3": [5, 11, 17, 10, 2, 4, 16, 20, 22, 6]},
    "F":  {"g1": [4, 8, 16, 9, 2, 15, 20, 10, 3, 7, 21, 17], "g2": [1, 19, 23, 5], "g3": []},
    "F'": {"g1": [], "g2": [3, 17, 21, 7], "g3": [4, 8, 16, 9, 2, 15, 20, 10, 1, 19, 23, 5]},
}

# Derive inverse face permutations from forward permutations
FACE_PERMS["U'"] = list(np.argsort(FACE_PERMS["U"]))
FACE_PERMS["L'"] = list(np.argsort(FACE_PERMS["L"]))
FACE_PERMS["F'"] = list(np.argsort(FACE_PERMS["F"]))

# Canonical move ordering (index i and i^1 are always inverses)
MOVE_LIST = ["U", "U'", "L", "L'", "F", "F'"]
NUM_MOVES = len(MOVE_LIST)


class MalteseGearCubeEnv:
    """
    GPU-accelerated Maltese Gear Cube environment.

    All state generation, expansion, and encoding happen on-device.
    Pre-allocated static buffers eliminate allocation overhead in hot loops.
    """

    def __init__(self):
        self.moves = MOVE_LIST
        self.num_moves = NUM_MOVES
        self.total_dim = 144

        # Goal state: 96 face colors + 24 gear positions + 24 gear values
        self.goal_colors = np.concatenate([
            np.repeat(np.arange(6), 16),                          # 96 face stickers
            [2, 2, 2, 2, 1, 4, 0, 5, 1, 1, 4, 4, 0, 0, 5, 5,   # 24 gear positions
             1, 4, 0, 5, 3, 3, 3, 3,
             0, 1, 2, 3, 0, 0, 0, 0, 1, 3, 1, 3, 1, 3, 1, 3,   # 24 gear values
             2, 2, 2, 2, 0, 1, 2, 3],
        ]).astype(np.uint8)

        # Precompute GPU-ready permutation indices and gear addends for each move
        self.idx_new, self.idx_old, self.adds = [], [], []
        for move in self.moves:
            idx_state = np.arange(self.total_dim, dtype=np.int32)[None, :]
            perm_res = self._apply_move_raw(idx_state, move, track_indices=True)

            zero_state = np.zeros((1, self.total_dim), dtype=np.int32)
            add_res = self._apply_move_raw(zero_state, move, track_indices=False)

            self.idx_new.append(np.arange(self.total_dim, dtype=int))
            self.idx_old.append(perm_res[0].astype(int))
            self.adds.append(add_res[0].astype(np.int8))

    # ----------------------------------------------------------------
    # GPU Setup
    # ----------------------------------------------------------------

    def setup_gpu(self, device, max_chunk_size):
        """Transfer precomputed tables to GPU and allocate static buffers."""
        self.device = device

        # Permutation and addend tables: shape (num_moves, 144)
        self.T_idx_old = torch.tensor(
            np.array(self.idx_old), dtype=torch.int64, device=device
        )
        self.T_adds = torch.tensor(
            np.array(self.adds), dtype=torch.int16, device=device
        )
        self.goal_colors_tensor = torch.tensor(
            self.goal_colors, dtype=torch.uint8, device=device
        )

        # --- STATIC MEMORY PRE-ALLOCATION ---
        # 144 positions: 120 colors (6 classes) + 24 gears (4 classes) = 720 + 96 = 816 dims
        self.static_nn_input = torch.zeros(
            (max_chunk_size, 816), dtype=torch.float16, device=device
        )
        self.color_offsets = torch.arange(0, 720, 6, device=device).unsqueeze(0)
        self.gear_offsets = torch.arange(720, 816, 4, device=device).unsqueeze(0)

    # ----------------------------------------------------------------
    # GPU Operations
    # ----------------------------------------------------------------

    def is_solved_gpu(self, states_tensor):
        """Check if each state in the batch matches the goal state."""
        return torch.all(
            states_tensor == self.goal_colors_tensor.unsqueeze(0), dim=1
        )

    def expand_gpu(self, states_tensor):
        """Expand each state into all possible children. Returns (B, num_moves, 144)."""
        B = states_tensor.size(0)
        states_expanded = states_tensor.unsqueeze(1).expand(B, self.num_moves, 144)
        idx_expanded = self.T_idx_old.unsqueeze(0).expand(B, self.num_moves, 144)
        adds_expanded = self.T_adds.unsqueeze(0).expand(B, self.num_moves, 144)

        children = torch.gather(states_expanded, 2, idx_expanded)

        # Gear values (positions 120–143): apply mod-4 addends
        gears = children[:, :, 120:].to(torch.int16)
        adds = adds_expanded[:, :, 120:]
        children[:, :, 120:] = torch.remainder(gears + adds, 4).to(torch.uint8)

        return children

    def generate_scrambled_states_gpu(
        self, num_states, max_moves, exact_moves=None, chunk_size=500_000
    ):
        """
        Generate scrambled states on GPU by backward-scrambling from goal.

        Depths sampled uniformly from {0, ..., max_moves} (or fixed via exact_moves).
        Inverse-move avoidance ensures no self-cancelling consecutive pairs.
        """
        all_states = torch.empty(
            (num_states, 144), dtype=torch.uint8, device=self.device
        )

        for offset in range(0, num_states, chunk_size):
            size = min(chunk_size, num_states - offset)
            states = self.goal_colors_tensor.unsqueeze(0).expand(size, 144).clone()

            if exact_moves is not None:
                target_lengths = torch.full(
                    (size,), exact_moves, device=self.device, dtype=torch.long
                )
            else:
                target_lengths = torch.randint(
                    0, max_moves + 1, (size,), device=self.device, dtype=torch.long
                )

            current_lengths = torch.zeros(size, device=self.device, dtype=torch.long)
            max_len = target_lengths.max().item() if size > 0 else 0
            prev_actions = torch.full(
                (size,), -1, device=self.device, dtype=torch.long
            )

            for _ in range(max_len):
                active_mask = current_lengths < target_lengths
                if not active_mask.any():
                    break

                # Random actions with inverse-move avoidance
                actions = torch.randint(
                    0, self.num_moves, (size,), device=self.device
                )
                has_prev = prev_actions != -1
                inverse_prev = prev_actions ^ 1
                rand_inc = torch.randint(
                    1, self.num_moves, (size,), device=self.device
                )
                new_actions = torch.remainder(
                    inverse_prev + rand_inc, self.num_moves
                )
                actions = torch.where(has_prev, new_actions, actions)
                prev_actions = torch.where(active_mask, actions, prev_actions)

                # Apply moves
                action_idx = self.T_idx_old[actions]
                next_states = torch.gather(states, 1, action_idx)

                gears = next_states[:, 120:].to(torch.int16)
                adds = self.T_adds[actions, 120:]
                next_states[:, 120:] = torch.remainder(gears + adds, 4).to(
                    torch.uint8
                )

                states = torch.where(
                    active_mask.unsqueeze(1), next_states, states
                )
                current_lengths += active_mask.long()

            all_states[offset : offset + size] = states

        return all_states

    # ----------------------------------------------------------------
    # State-to-Network-Input Encoding
    # ----------------------------------------------------------------

    def states_to_nnet_input_static(self, states_tensor):
        """
        Zero-allocation one-hot encoding using pre-allocated static buffer.

        Used during training for maximum throughput. Requires batch size
        <= max_chunk_size passed to setup_gpu().
        """
        B = states_tensor.size(0)
        out_buffer = self.static_nn_input[:B]
        out_buffer.zero_()

        colors = states_tensor[:, :120].long()
        color_scatter_idx = colors + self.color_offsets
        out_buffer.scatter_(1, color_scatter_idx, 1.0)

        gears = states_tensor[:, 120:].long()
        gear_scatter_idx = gears + self.gear_offsets
        out_buffer.scatter_(1, gear_scatter_idx, 1.0)

        return out_buffer

    def states_to_nnet_input(self, states_tensor):
        """
        Dynamic one-hot encoding (allocates fresh memory).

        Used during search/evaluation where batch sizes vary dynamically.
        """
        colors = states_tensor[:, :120].long()
        gears = states_tensor[:, 120:].long()
        colors_oh = F.one_hot(colors, num_classes=6).view(
            states_tensor.size(0), -1
        ).half()
        gears_oh = F.one_hot(gears, num_classes=4).view(
            states_tensor.size(0), -1
        ).half()
        return torch.cat([colors_oh, gears_oh], dim=1)

    # ----------------------------------------------------------------
    # Raw Move Application (CPU, used only during __init__)
    # ----------------------------------------------------------------

    def _apply_move_raw(self, states_np, action_str, track_indices=True):
        """
        Apply a single move to a batch of states on CPU.

        Uses dictionary lookups for permutations and gear increments.
        Called only during __init__ to precompute GPU tables.
        """
        states_next = states_np.copy()

        # 1. Face sticker permutation (positions 0–95)
        face_perm = FACE_PERMS[action_str]
        states_next[:, :96] = states_np[:, :96][:, face_perm]

        # 2. Gear position permutation (positions 96–119)
        gear_pos_perm = GEAR_POS_PERMS[action_str]
        states_next[:, 96:120] = states_np[:, 96:120][:, gear_pos_perm]

        # 3. Gear value increments (positions 120–143)
        increments = GEAR_INCREMENTS[action_str]
        p3_vals = states_np[:, 120:144].copy()

        if not track_indices:
            g1 = increments["g1"]
            g2 = increments["g2"]
            g3 = increments["g3"]
            if g1:
                p3_vals[:, g1] = (p3_vals[:, g1] + 1) % 4
            if g2:
                p3_vals[:, g2] = (p3_vals[:, g2] + 2) % 4
            if g3:
                p3_vals[:, g3] = (p3_vals[:, g3] + 3) % 4

        states_next[:, 120:144] = p3_vals[:, gear_pos_perm]

        return states_next
