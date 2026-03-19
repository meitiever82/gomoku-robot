"""
五子棋 AI — 启发式评分 + Alpha-Beta 搜索
"""

import numpy as np
from .gomoku_engine import GomokuEngine, EMPTY, BLACK, WHITE, DIRECTIONS
from .patterns import THREAT_PATTERNS, SCORE_FIVE


def _opponent(color: int) -> int:
    return WHITE if color == BLACK else BLACK


class GomokuAI:
    def __init__(self, board_size: int = 9, max_depth: int = 4):
        self.board_size = board_size
        self.max_depth = max_depth
        self._engine = GomokuEngine(board_size)

    def get_best_move(self, board: np.ndarray, color: int) -> tuple[int, int]:
        """计算最佳落子位置"""
        candidates = self._engine.get_neighbor_positions(board, radius=2)

        if len(candidates) == 1:
            return candidates[0]

        # 先检查是否有立即获胜的棋
        for r, c in candidates:
            board[r, c] = color
            if self._engine.check_winner(board) == color:
                board[r, c] = EMPTY
                return (r, c)
            board[r, c] = EMPTY

        # 检查是否需要防守对方的五连
        opp = _opponent(color)
        for r, c in candidates:
            board[r, c] = opp
            if self._engine.check_winner(board) == opp:
                board[r, c] = EMPTY
                return (r, c)
            board[r, c] = EMPTY

        # Alpha-Beta 搜索
        best_score = -float("inf")
        best_move = candidates[0]

        # 按启发式评分排序候选位置（提高剪枝效率）
        scored = []
        for r, c in candidates:
            board[r, c] = color
            s = self._evaluate_position(board, color)
            board[r, c] = EMPTY
            scored.append((s, r, c))
        scored.sort(reverse=True)

        # 限制搜索宽度
        max_candidates = min(15, len(scored))
        scored = scored[:max_candidates]

        for _, r, c in scored:
            board[r, c] = color
            score = self._alphabeta(
                board, self.max_depth - 1, -float("inf"), float("inf"), False, color
            )
            board[r, c] = EMPTY

            if score > best_score:
                best_score = score
                best_move = (r, c)

        return best_move

    def _alphabeta(
        self,
        board: np.ndarray,
        depth: int,
        alpha: float,
        beta: float,
        is_maximizing: bool,
        ai_color: int,
    ) -> float:
        """Alpha-Beta 剪枝搜索"""
        winner = self._engine.check_winner(board)
        if winner == ai_color:
            return SCORE_FIVE + depth  # 越早赢越好
        if winner == _opponent(ai_color):
            return -(SCORE_FIVE + depth)
        if depth == 0 or self._engine.is_board_full(board):
            return self._evaluate_board(board, ai_color)

        candidates = self._engine.get_neighbor_positions(board, radius=2)
        # 限制搜索宽度
        if len(candidates) > 10:
            scored = []
            eval_color = ai_color if is_maximizing else _opponent(ai_color)
            for r, c in candidates:
                board[r, c] = eval_color
                s = self._evaluate_position(board, eval_color)
                board[r, c] = EMPTY
                scored.append((s, r, c))
            scored.sort(reverse=True)
            candidates = [(r, c) for _, r, c in scored[:10]]

        current_color = ai_color if is_maximizing else _opponent(ai_color)

        if is_maximizing:
            max_eval = -float("inf")
            for r, c in candidates:
                board[r, c] = current_color
                eval_score = self._alphabeta(board, depth - 1, alpha, beta, False, ai_color)
                board[r, c] = EMPTY
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float("inf")
            for r, c in candidates:
                board[r, c] = current_color
                eval_score = self._alphabeta(board, depth - 1, alpha, beta, True, ai_color)
                board[r, c] = EMPTY
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def _evaluate_board(self, board: np.ndarray, ai_color: int) -> float:
        """评估整个棋盘对 ai_color 的优势"""
        ai_score = self._evaluate_position(board, ai_color)
        opp_score = self._evaluate_position(board, _opponent(ai_color))
        return ai_score - opp_score * 1.1  # 防守略高权重

    def _evaluate_position(self, board: np.ndarray, color: int) -> float:
        """评估棋盘上某一方的模式得分"""
        score = 0.0
        size = board.shape[0]
        opp = _opponent(color)

        # 在四个方向上提取所有线段并匹配模式
        lines = self._extract_lines(board, color, opp)
        for line in lines:
            score += self._score_line(line)

        return score

    def _extract_lines(
        self, board: np.ndarray, color: int, opp: int
    ) -> list[str]:
        """提取所有方向上的线段字符串"""
        size = board.shape[0]
        lines = []
        symbol_map = {EMPTY: "_", color: "X", opp: "O"}

        # 横向
        for r in range(size):
            line = "".join(symbol_map[board[r, c]] for c in range(size))
            lines.append(line)

        # 纵向
        for c in range(size):
            line = "".join(symbol_map[board[r, c]] for r in range(size))
            lines.append(line)

        # 主对角线 (↘)
        for start in range(-(size - 1), size):
            line = []
            for i in range(size):
                r, c = i, i - start
                if 0 <= r < size and 0 <= c < size:
                    line.append(symbol_map[board[r, c]])
            if len(line) >= 5:
                lines.append("".join(line))

        # 副对角线 (↙)
        for start in range(0, 2 * size - 1):
            line = []
            for i in range(size):
                r, c = i, start - i
                if 0 <= r < size and 0 <= c < size:
                    line.append(symbol_map[board[r, c]])
            if len(line) >= 5:
                lines.append("".join(line))

        return lines

    def _score_line(self, line: str) -> float:
        """对一条线段匹配所有威胁模式并计分"""
        score = 0.0
        for pattern, pattern_score in THREAT_PATTERNS:
            idx = 0
            while True:
                pos = line.find(pattern, idx)
                if pos == -1:
                    break
                score += pattern_score
                idx = pos + 1
        return score
