"""
机器人标定 — 棋盘坐标 → 机器人基坐标的刚体变换

使用方法:
  gomoku-calibrate-robot
  或:
  python -m gomoku_robot.calibration.calibrate_robot

标定流程:
1. 用 leader 臂示教，依次触碰棋盘上 4 个已知交叉点
2. 记录每个点的关节角 → FK → 末端位置
3. 求解棋盘→机器人的刚体变换 (R, t)
4. 同样标定托盘的起始位置和方向
"""

import numpy as np
from ..config import GomokuRobotConfig, CALIBRATION_DIR
from .transforms import CoordinateTransformer


def solve_rigid_transform(
    points_board: np.ndarray, points_robot: np.ndarray
) -> np.ndarray:
    """求解棋盘→机器人坐标的刚体变换 (最小二乘)

    Args:
        points_board: 棋盘坐标点 (N, 3)，N >= 3
        points_robot: 对应的机器人坐标点 (N, 3)

    Returns:
        4×4 齐次变换矩阵
    """
    assert points_board.shape == points_robot.shape
    assert points_board.shape[0] >= 3

    # 质心
    centroid_b = points_board.mean(axis=0)
    centroid_r = points_robot.mean(axis=0)

    # 去质心
    B = points_board - centroid_b
    R_pts = points_robot - centroid_r

    # SVD 求旋转
    H = B.T @ R_pts
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 确保是正交旋转 (det = +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 平移
    t = centroid_r - R @ centroid_b

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def interactive_calibration():
    """交互式标定 — 手动输入坐标点对"""
    cfg = GomokuRobotConfig()
    transformer = CoordinateTransformer(cfg.board, cfg.tray)

    # 尝试加载已有的像素→棋盘标定
    try:
        transformer.load()
        print("已加载像素→棋盘标定数据")
    except FileNotFoundError:
        print("警告: 未找到像素→棋盘标定数据，请先运行 calibrate_board")

    print("\n=== 棋盘→机器人标定 ===")
    print("请用 leader 臂触碰以下棋盘交叉点，并输入末端执行器坐标 (mm)")
    print("格式: x y z (空格分隔)\n")

    n = cfg.board.size - 1
    # 标定用的棋盘点: 四个角 + 中心
    calib_points = [
        (0, 0, "左上"),
        (0, n, "右上"),
        (n, n, "右下"),
        (n, 0, "左下"),
    ]

    points_board = []
    points_robot = []

    for row, col, name in calib_points:
        board_x = col * cfg.board.grid_spacing_mm
        board_y = row * cfg.board.grid_spacing_mm
        board_z = 0.0

        while True:
            raw = input(f"  触碰 ({row},{col}) {name}，输入 EE 坐标 (x y z mm): ").strip()
            if raw.lower() == "q":
                print("取消标定")
                return
            try:
                x, y, z = map(float, raw.split())
                break
            except ValueError:
                print("  格式错误，请输入三个数字，如: 150.0 -80.0 25.0")

        points_board.append([board_x, board_y, board_z])
        points_robot.append([x, y, z])

    points_board = np.array(points_board)
    points_robot = np.array(points_robot)

    T = solve_rigid_transform(points_board, points_robot)
    transformer.T_board_to_robot = T

    # 验证
    print("\n标定验证:")
    for i, (row, col, name) in enumerate(calib_points):
        computed = transformer.board_to_robot(row, col)
        actual = points_robot[i]
        error = np.linalg.norm(computed - actual)
        print(f"  {name}: 误差 = {error:.2f} mm")

    print("\n=== 托盘标定 ===")
    print("请触碰托盘的第一个槽位和最后一个槽位")

    raw = input("  第一个槽位 EE 坐标 (x y z mm): ").strip()
    tray_start = np.array(list(map(float, raw.split())))

    raw = input("  最后一个槽位 EE 坐标 (x y z mm): ").strip()
    tray_end = np.array(list(map(float, raw.split())))

    transformer.tray_origin_robot = tray_start
    direction = tray_end - tray_start
    transformer.tray_direction_robot = direction / np.linalg.norm(direction)

    transformer.save()
    print(f"\n标定数据已保存到 {CALIBRATION_DIR}/transforms.json")


def main():
    interactive_calibration()


if __name__ == "__main__":
    main()
