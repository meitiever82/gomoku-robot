"""五子棋引擎单元测试"""

import numpy as np
import pytest
from gomoku_robot.engine.gomoku_engine import GomokuEngine, BLACK, WHITE, EMPTY
from gomoku_robot.engine.ai_player import GomokuAI


class TestGomokuEngine:
    def setup_method(self):
        self.engine = GomokuEngine(board_size=9)

    def test_empty_board(self):
        assert self.engine.check_winner() is None
        assert self.engine.get_game_state() == "playing"
        assert not self.engine.is_board_full()

    def test_valid_move(self):
        assert self.engine.place(0, 0, BLACK)
        assert self.engine.board[0, 0] == BLACK
        assert not self.engine.is_valid_move(0, 0)  # 已占用

    def test_invalid_move(self):
        self.engine.place(0, 0, BLACK)
        assert not self.engine.place(0, 0, WHITE)  # 重复落子
        assert not self.engine.place(-1, 0, BLACK)  # 越界
        assert not self.engine.place(0, 9, BLACK)   # 越界

    def test_undo(self):
        self.engine.place(4, 4, BLACK)
        assert self.engine.board[4, 4] == BLACK
        self.engine.undo()
        assert self.engine.board[4, 4] == EMPTY

    def test_horizontal_win(self):
        for c in range(5):
            self.engine.place(0, c, BLACK)
        assert self.engine.check_winner() == BLACK
        assert self.engine.get_game_state() == "black_wins"

    def test_vertical_win(self):
        for r in range(5):
            self.engine.place(r, 0, WHITE)
        assert self.engine.check_winner() == WHITE

    def test_diagonal_win(self):
        for i in range(5):
            self.engine.place(i, i, BLACK)
        assert self.engine.check_winner() == BLACK

    def test_anti_diagonal_win(self):
        for i in range(5):
            self.engine.place(i, 4 - i, WHITE)
        assert self.engine.check_winner() == WHITE

    def test_no_winner_four(self):
        """四连不算赢"""
        for c in range(4):
            self.engine.place(0, c, BLACK)
        assert self.engine.check_winner() is None

    def test_neighbor_positions(self):
        self.engine.place(4, 4, BLACK)
        neighbors = self.engine.get_neighbor_positions(radius=1)
        assert (4, 4) not in neighbors  # 已占用的不在列表中
        assert (3, 3) in neighbors
        assert (5, 5) in neighbors
        assert len(neighbors) == 8  # 3×3 - 1

    def test_empty_board_center(self):
        """空棋盘应返回中心点"""
        positions = self.engine.get_neighbor_positions()
        assert positions == [(4, 4)]

    def test_print_board(self, capsys):
        self.engine.place(4, 4, BLACK)
        self.engine.place(4, 5, WHITE)
        self.engine.print_board()
        captured = capsys.readouterr()
        assert "●" in captured.out
        assert "○" in captured.out


class TestGomokuAI:
    def setup_method(self):
        self.ai = GomokuAI(board_size=9, max_depth=2)

    def test_win_move(self):
        """AI 应该抓住必赢的一步"""
        board = np.zeros((9, 9), dtype=np.int8)
        # 黑方四连，差一步赢
        for c in range(4):
            board[4, c] = BLACK
        move = self.ai.get_best_move(board, BLACK)
        assert move == (4, 4), f"应落在 (4,4) 完成五连，实际 {move}"

    def test_block_opponent_win(self):
        """AI 应该堵住对方的四连"""
        board = np.zeros((9, 9), dtype=np.int8)
        # 白方四连
        for c in range(4):
            board[4, c] = WHITE
        move = self.ai.get_best_move(board, BLACK)
        assert move == (4, 4), f"应堵在 (4,4)，实际 {move}"

    def test_first_move_center(self):
        """空棋盘首步应在中心附近"""
        board = np.zeros((9, 9), dtype=np.int8)
        move = self.ai.get_best_move(board, BLACK)
        assert move == (4, 4)

    def test_ai_does_not_crash(self):
        """AI 在复杂棋局中不崩溃"""
        board = np.zeros((9, 9), dtype=np.int8)
        # 随机放一些棋子
        rng = np.random.default_rng(42)
        for _ in range(20):
            r, c = rng.integers(0, 9, size=2)
            if board[r, c] == EMPTY:
                board[r, c] = rng.choice([BLACK, WHITE])
        move = self.ai.get_best_move(board, BLACK)
        assert 0 <= move[0] < 9 and 0 <= move[1] < 9
        assert board[move[0], move[1]] == EMPTY


class TestTransforms:
    def test_rigid_transform_identity(self):
        """相同点应得到近似单位变换"""
        from gomoku_robot.calibration.calibrate_robot import solve_rigid_transform

        pts = np.array([
            [0, 0, 0],
            [100, 0, 0],
            [100, 100, 0],
            [0, 100, 0],
        ], dtype=np.float64)

        T = solve_rigid_transform(pts, pts)
        np.testing.assert_allclose(T, np.eye(4), atol=1e-10)

    def test_rigid_transform_translation(self):
        """纯平移"""
        from gomoku_robot.calibration.calibrate_robot import solve_rigid_transform

        pts_a = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
        ], dtype=np.float64)

        offset = np.array([10, 20, 30])
        pts_b = pts_a + offset

        T = solve_rigid_transform(pts_a, pts_b)
        np.testing.assert_allclose(T[:3, 3], offset, atol=1e-10)
        np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
