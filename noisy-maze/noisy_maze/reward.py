"""One true-maze validator shared by SFT evaluation and the custom RL reward."""

from collections import deque

ACTION_MAP = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}


def parse_ground_truth(sequence: str) -> tuple[list[list[int]], tuple[int, int], tuple[int, int]]:
    tokens = sequence.split()
    first, last = tokens.index("GRID_START"), tokens.index("GRID_END")
    rows, row = [], []
    starts, goals = [], []
    for token in tokens[first + 1 : last]:
        if token == "NEWLINE":
            if not row:
                raise ValueError("Empty ground-truth grid row")
            rows.append(row)
            row = []
        elif token in {"WALL", "PATH", "START", "GOAL"}:
            if token == "START":
                starts.append((len(rows), len(row)))
            elif token == "GOAL":
                goals.append((len(rows), len(row)))
            row.append(int(token == "WALL"))
        else:
            raise ValueError(f"Ground truth must be unclouded; invalid grid token: {token}")
    if row:
        rows.append(row)
    if not rows or any(len(row) != len(rows) for row in rows) or len(starts) != 1 or len(goals) != 1:
        raise ValueError("Ground truth must be square with exactly one START and one GOAL")
    return rows, starts[0], goals[0]


def parse_actions(solution: str) -> tuple[list[str], str]:
    tokens = solution.split()
    if "DONE" not in tokens:
        return [], "missing_done"
    actions = tokens[: tokens.index("DONE")]
    if not actions:
        return [], "no_actions"
    if any(action not in ACTION_MAP for action in actions):
        return [], "invalid_action"
    return actions, "success"


def validate_solution(solution: str, ground_truth: str) -> tuple[bool, str]:
    # A malformed or clouded reference is a dataset error, not a failed model response.
    grid, current, goal = parse_ground_truth(ground_truth)
    actions, reason = parse_actions(solution)
    if reason != "success":
        return False, reason
    for action in actions:
        dr, dc = ACTION_MAP[action]
        r, c = current[0] + dr, current[1] + dc
        if not (0 <= r < len(grid) and 0 <= c < len(grid)):
            return False, "out_of_bounds"
        if grid[r][c]:
            return False, "hit_wall"
        current = (r, c)
        # Match the original maze environment's terminal-on-reaching-goal semantics.
        if current == goal:
            return True, "success"
    return False, "not_at_goal"


def compute_optimal_length(ground_truth: str) -> int:
    grid, start, goal = parse_ground_truth(ground_truth)
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        (r, c), distance = queue.popleft()
        if (r, c) == goal:
            return distance
        for dr, dc in ACTION_MAP.values():
            nxt = (r + dr, c + dc)
            if 0 <= nxt[0] < len(grid) and 0 <= nxt[1] < len(grid) and not grid[nxt[0]][nxt[1]] and nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, distance + 1))
    raise ValueError("Ground-truth maze is unsolvable")


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    if not data_source.startswith("noisy_maze_"):
        raise ValueError(f"Unexpected data source: {data_source}")
    success, _ = validate_solution(solution_str, ground_truth)
    return float(success)


def compute_scores(data_sources, solution_strs, ground_truths, extra_infos) -> list[float]:
    """Adapter for verl's batch manager, avoiding custom-function multiprocessing pickling."""
    if len({len(data_sources), len(solution_strs), len(ground_truths), len(extra_infos)}) != 1:
        raise ValueError("Reward batch columns have different lengths")
    return [compute_score(source, solution, truth, extra) for source, solution, truth, extra in zip(data_sources, solution_strs, ground_truths, extra_infos, strict=True)]
