def is_safe(board, row, col, n, blocked):
    if (row, col) in blocked:
        return False
    for i in range(row):
        if board[i] == col or abs(board[i] - col) == abs(i - row):
            return False
    return True

def solve_n_queens_with_blocks(n, blocked):
    board = [-1] * n
    solutions = []
    def backtrack(row):
        if row == n:
            solutions.append([(i, board[i]) for i in range(n)])
            return
        for col in range(n):
            if is_safe(board, row, col, n, blocked):
                board[row] = col
                backtrack(row + 1)
                board[row] = -1
    backtrack(0)
    return solutions

n = 5
blocked_cells = [(0, 2), (2, 3), (3, 0)]
solutions = solve_n_queens_with_blocks(n, blocked_cells)

for idx, sol in enumerate(solutions, start=1):
    print(f"Solution {idx}: {sol}")

print("\nTotal Solutions:", len(solutions))
