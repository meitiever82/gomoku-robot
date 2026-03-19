"""
棋盘状态检测 — 从相机图像提取 9×9 棋盘状态
"""

import cv2
import numpy as np
from ..config import GomokuRobotConfig, VisionConfig
from ..calibration.transforms import CoordinateTransformer


EMPTY = 0
BLACK = 1
WHITE = 2


class BoardDetector:
    """棋盘状态检测器

    流程: 拍照 → ArUco/单应性校正 → 采样交叉点 → HSV 分类
    """

    def __init__(self, cfg: GomokuRobotConfig, transformer: CoordinateTransformer):
        self.cfg = cfg
        self.transformer = transformer
        self.board_size = cfg.board.size
        self.vcfg = cfg.vision

        self._cap: cv2.VideoCapture | None = None

    def connect(self):
        """打开相机"""
        self._cap = cv2.VideoCapture(self.cfg.camera.index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera.height)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开相机 {self.cfg.camera.index}")

    def disconnect(self):
        """关闭相机"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def capture_frame(self) -> np.ndarray:
        """拍一帧"""
        if self._cap is None:
            raise RuntimeError("相机未连接")
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("相机读取失败")
        return frame

    def capture_averaged(self) -> np.ndarray:
        """多帧平均以减少噪声"""
        frames = []
        for _ in range(self.vcfg.num_frames_avg):
            frames.append(self.capture_frame().astype(np.float32))
        avg = np.mean(frames, axis=0).astype(np.uint8)
        return avg

    def detect_board(self, frame: np.ndarray | None = None) -> np.ndarray:
        """检测棋盘状态

        Args:
            frame: 输入图像，None 则自动拍照

        Returns:
            np.ndarray shape (board_size, board_size), dtype=int8
            0=空, 1=黑, 2=白
        """
        if frame is None:
            frame = self.capture_averaged()

        board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        patch_r = self.vcfg.patch_size // 2

        for r in range(self.board_size):
            for c in range(self.board_size):
                u, v = self.transformer.board_to_pixel(r, c)
                u, v = int(round(u)), int(round(v))

                # 边界检查
                y1 = max(0, v - patch_r)
                y2 = min(frame.shape[0], v + patch_r + 1)
                x1 = max(0, u - patch_r)
                x2 = min(frame.shape[1], u + patch_r + 1)

                if y2 <= y1 or x2 <= x1:
                    continue

                patch = hsv[y1:y2, x1:x2]
                board[r, c] = self._classify_patch(patch)

        return board

    def _classify_patch(self, patch_hsv: np.ndarray) -> int:
        """根据 HSV 值分类一个 patch

        主要依据 V (明度) 通道:
        - 黑子: V 低
        - 白子: V 高
        - 空位: V 中间 (棋盘底色)
        """
        v_channel = patch_hsv[:, :, 2]
        mean_v = np.mean(v_channel)
        # 也检查饱和度 - 棋子通常低饱和度
        s_channel = patch_hsv[:, :, 1]
        mean_s = np.mean(s_channel)

        if mean_v < self.vcfg.black_v_max:
            return BLACK
        elif mean_v > self.vcfg.white_v_min:
            return WHITE
        else:
            return EMPTY

    def detect_and_visualize(self, frame: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """检测棋盘并返回可视化图像

        Returns:
            (board_state, visualization_image)
        """
        if frame is None:
            frame = self.capture_averaged()

        board = self.detect_board(frame)
        vis = frame.copy()

        for r in range(self.board_size):
            for c in range(self.board_size):
                u, v = self.transformer.board_to_pixel(r, c)
                u, v = int(round(u)), int(round(v))
                state = board[r, c]

                if state == BLACK:
                    cv2.circle(vis, (u, v), 8, (0, 0, 0), -1)
                    cv2.circle(vis, (u, v), 8, (0, 255, 0), 1)
                elif state == WHITE:
                    cv2.circle(vis, (u, v), 8, (255, 255, 255), -1)
                    cv2.circle(vis, (u, v), 8, (0, 255, 0), 1)
                else:
                    cv2.circle(vis, (u, v), 3, (0, 255, 0), 1)

        return board, vis
