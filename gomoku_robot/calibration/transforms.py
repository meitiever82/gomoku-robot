"""
坐标变换工具 — 像素 ↔ 棋盘 ↔ 机器人坐标
"""

import json
from pathlib import Path
import numpy as np
from ..config import BoardConfig, TrayConfig, CALIBRATION_DIR


class CoordinateTransformer:
    """管理三个坐标系之间的变换:
    1. 像素坐标 (u, v) — 相机图像
    2. 棋盘坐标 (row, col) — 棋盘交叉点索引
    3. 机器人坐标 (x, y, z) — 机器人基坐标系 (mm)
    """

    def __init__(self, board_cfg: BoardConfig, tray_cfg: TrayConfig):
        self.board_cfg = board_cfg
        self.tray_cfg = tray_cfg

        # 像素→棋盘的单应性矩阵 (3×3)，由 calibrate_board 计算
        self.H_pixel_to_board: np.ndarray | None = None

        # 棋盘→机器人的刚体变换 (4×4)，由 calibrate_robot 计算
        self.T_board_to_robot: np.ndarray | None = None

        # 托盘原点在机器人坐标系中的位置 (3,)
        self.tray_origin_robot: np.ndarray | None = None
        # 托盘排列方向在机器人坐标系中的单位向量 (3,)
        self.tray_direction_robot: np.ndarray | None = None

    # ---- 像素 → 棋盘 ----

    def pixel_to_board_continuous(self, u: float, v: float) -> tuple[float, float]:
        """像素坐标 → 连续棋盘坐标 (可以是小数)"""
        if self.H_pixel_to_board is None:
            raise RuntimeError("未标定: 请先运行 calibrate_board")
        pt = np.array([u, v, 1.0])
        result = self.H_pixel_to_board @ pt
        result /= result[2]
        return float(result[0]), float(result[1])

    def pixel_to_board(self, u: float, v: float) -> tuple[int, int]:
        """像素坐标 → 最近的棋盘交叉点 (row, col)"""
        bx, by = self.pixel_to_board_continuous(u, v)
        row = int(round(by))
        col = int(round(bx))
        row = max(0, min(self.board_cfg.size - 1, row))
        col = max(0, min(self.board_cfg.size - 1, col))
        return row, col

    # ---- 棋盘 → 机器人 ----

    def board_to_robot(self, row: int, col: int) -> np.ndarray:
        """棋盘交叉点 → 机器人坐标 (x, y, z) mm"""
        if self.T_board_to_robot is None:
            raise RuntimeError("未标定: 请先运行 calibrate_robot")

        # 棋盘物理坐标 (mm)，原点在 (0,0) 交叉点
        board_x = col * self.board_cfg.grid_spacing_mm
        board_y = row * self.board_cfg.grid_spacing_mm
        board_z = 0.0  # 棋盘平面

        pt_board = np.array([board_x, board_y, board_z, 1.0])
        pt_robot = self.T_board_to_robot @ pt_board
        return pt_robot[:3]

    # ---- 托盘 → 机器人 ----

    def tray_slot_to_robot(self, slot_index: int) -> np.ndarray:
        """托盘槽位 → 机器人坐标 (x, y, z) mm"""
        if self.tray_origin_robot is None or self.tray_direction_robot is None:
            raise RuntimeError("未标定: 请先设置托盘位置")

        offset = slot_index * self.tray_cfg.slot_spacing_mm
        return self.tray_origin_robot + offset * self.tray_direction_robot

    # ---- 棋盘交叉点 → 像素 (反向，用于可视化) ----

    def board_to_pixel(self, row: int, col: int) -> tuple[float, float]:
        """棋盘交叉点 → 像素坐标"""
        if self.H_pixel_to_board is None:
            raise RuntimeError("未标定: 请先运行 calibrate_board")
        H_inv = np.linalg.inv(self.H_pixel_to_board)
        # 棋盘连续坐标: col 是 x, row 是 y
        pt = np.array([float(col), float(row), 1.0])
        result = H_inv @ pt
        result /= result[2]
        return float(result[0]), float(result[1])

    # ---- 保存/加载 ----

    def save(self, path: Path | None = None):
        """保存标定数据到 JSON"""
        if path is None:
            path = CALIBRATION_DIR / "transforms.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {}
        if self.H_pixel_to_board is not None:
            data["H_pixel_to_board"] = self.H_pixel_to_board.tolist()
        if self.T_board_to_robot is not None:
            data["T_board_to_robot"] = self.T_board_to_robot.tolist()
        if self.tray_origin_robot is not None:
            data["tray_origin_robot"] = self.tray_origin_robot.tolist()
        if self.tray_direction_robot is not None:
            data["tray_direction_robot"] = self.tray_direction_robot.tolist()

        path.write_text(json.dumps(data, indent=2))

    def load(self, path: Path | None = None):
        """从 JSON 加载标定数据"""
        if path is None:
            path = CALIBRATION_DIR / "transforms.json"
        if not path.exists():
            raise FileNotFoundError(f"标定文件不存在: {path}")

        data = json.loads(path.read_text())

        if "H_pixel_to_board" in data:
            self.H_pixel_to_board = np.array(data["H_pixel_to_board"])
        if "T_board_to_robot" in data:
            self.T_board_to_robot = np.array(data["T_board_to_robot"])
        if "tray_origin_robot" in data:
            self.tray_origin_robot = np.array(data["tray_origin_robot"])
        if "tray_direction_robot" in data:
            self.tray_direction_robot = np.array(data["tray_direction_robot"])
