"""
五子棋引擎 — 规则、胜负判定、棋盘管理
"""

import numpy as np

EMPTY = 0
BLACK = 1
WHITE = 2

# 四个搜索方向: 横、竖、主对角线、副对角线
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


class GomokuEngine:
    def __init__(self, board_size: int = 9):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.move_history: list[tuple[int, int, int]] = []  # (row, col, color)

    def reset(self):
        self.board.fill(EMPTY)
        self.move_history.clear()

    def is_valid_move(self, row: int, col: int) -> bool:
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return False
        return self.board[row, col] == EMPTY

    def place(self, row: int, col: int, color: int) -> bool:
        """落子，返回是否成功"""
        if not self.is_valid_move(row, col):
            return False
        self.board[row, col] = color
        self.move_history.append((row, col, color))
        return True

    def undo(self) -> bool:
        """悔棋"""
        if not self.move_history:
            return False
        row, col, _ = self.move_history.pop()
        self.board[row, col] = EMPTY
        return True

    def check_winner(self, board: np.ndarray | None = None) -> int | None:
        """检查是否有人获胜，返回获胜方颜色或 None"""
        if board is None:
            board = self.board
        size = board.shape[0]

        for r in range(size):
            for c in range(size):
                color = board[r, c]
                if color == EMPTY:
                    continue
                for dr, dc in DIRECTIONS:
                    if self._count_consecutive(board, r, c, dr, dc, color) >= 5:
                        return color
        return None

    def _count_consecutive(
        self, board: np.ndarray, r: int, c: int, dr: int, dc: int, color: int
    ) -> int:
        """沿方向计数连续同色棋子"""
        count = 0
        size = board.shape[0]
        while 0 <= r < size and 0 <= c < size and board[r, c] == color:
            count += 1
            r += dr
            c += dc
        return count

    def is_board_full(self, board: np.ndarray | None = None) -> bool:
        if board is None:
            board = self.board
        return not np.any(board == EMPTY)

    def get_game_state(self, board: np.ndarray | None = None) -> str:
        """返回游戏状态: 'playing', 'black_wins', 'white_wins', 'draw'"""
        if board is None:
            board = self.board
        winner = self.check_winner(board)
        if winner == BLACK:
            return "black_wins"
        if winner == WHITE:
            return "white_wins"
        if self.is_board_full(board):
            return "draw"
        return "playing"

    def get_empty_positions(self, board: np.ndarray | None = None) -> list[tuple[int, int]]:
        if board is None:
            board = self.board
        positions = []
        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                if board[r, c] == EMPTY:
                    positions.append((r, c))
        return positions

    def get_neighbor_positions(
        self, board: np.ndarray | None = None, radius: int = 2
    ) -> list[tuple[int, int]]:
        """获取已有棋子附近的空位（缩小搜索范围）"""
        if board is None:
            board = self.board
        size = board.shape[0]
        candidates = set()

        for r in range(size):
            for c in range(size):
                if board[r, c] != EMPTY:
                    for dr in range(-radius, radius + 1):
                        for dc in range(-radius, radius + 1):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < size and 0 <= nc < size and board[nr, nc] == EMPTY:
                                candidates.add((nr, nc))

        if not candidates:
            # 空棋盘，返回中心
            center = size // 2
            return [(center, center)]

        return list(candidates)

    def print_board(self, board: np.ndarray | None = None):
        """打印棋盘到终端"""
        if board is None:
            board = self.board
        size = board.shape[0]
        symbols = {EMPTY: "·", BLACK: "●", WHITE: "○"}

        # 列号
        header = "   " + " ".join(f"{c:2d}" for c in range(size))
        print(header)

        for r in range(size):
            row_str = f"{r:2d} " + "  ".join(symbols[board[r, c]] for c in range(size))
            print(row_str)
        print()
