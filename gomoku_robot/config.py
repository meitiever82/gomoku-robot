"""五子棋机器人全局配置"""

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(__file__).parent.parent
CALIBRATION_DIR = PROJECT_ROOT / "calibration_data"


@dataclass
class BoardConfig:
    """棋盘配置"""
    size: int = 9                    # 棋盘大小 (9×9)
    grid_spacing_mm: float = 22.0    # 网格间距 (mm)
    piece_diameter_mm: float = 20.0  # 棋子直径 (mm)
    piece_height_mm: float = 6.0     # 棋子厚度 (mm)

    @property
    def board_size_mm(self) -> float:
        """棋盘物理尺寸 (mm)"""
        return (self.size - 1) * self.grid_spacing_mm


@dataclass
class CameraConfig:
    """相机配置"""
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class ArUcoConfig:
    """ArUco 标记配置"""
    dictionary: int = 0              # cv2.aruco.DICT_4X4_50
    marker_id: int = 0
    marker_size_mm: float = 30.0     # 标记物理尺寸 (mm)


@dataclass
class TrayConfig:
    """棋子托盘配置"""
    num_slots: int = 41              # 槽位数量 (9×9/2 + 1)
    slot_spacing_mm: float = 24.0    # 槽位间距 (mm)


@dataclass
class VisionConfig:
    """视觉检测配置"""
    patch_size: int = 15             # 交叉点采样 patch 大小 (像素)
    # HSV 阈值 - 需要根据实际环境调整
    black_v_max: int = 80            # 黑子 V 通道上限
    white_v_min: int = 180           # 白子 V 通道下限
    num_frames_avg: int = 3          # 多帧平均数


@dataclass
class RobotConfig:
    """机器人配置"""
    follower_port: str = "/dev/ttyACM0"
    leader_port: str = "/dev/ttyACM1"
    robot_id: str = "gomoku_follower"


@dataclass
class GomokuRobotConfig:
    """顶层配置"""
    board: BoardConfig = field(default_factory=BoardConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    aruco: ArUcoConfig = field(default_factory=ArUcoConfig)
    tray: TrayConfig = field(default_factory=TrayConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)

    # 棋子颜色定义
    EMPTY: int = 0
    BLACK: int = 1   # 机器人
    WHITE: int = 2   # 人类
