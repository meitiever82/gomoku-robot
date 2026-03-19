"""
棋盘标定 — ArUco 检测 → 像素到棋盘坐标的单应性矩阵

使用方法:
  gomoku-calibrate-board
  或:
  python -m gomoku_robot.calibration.calibrate_board

标定流程:
1. 在棋盘四个角放置 ArUco 标记（或手动点击四个角点）
2. 计算像素→棋盘坐标的单应性矩阵
3. 保存标定结果
"""

import cv2
import numpy as np
from ..config import GomokuRobotConfig, CALIBRATION_DIR
from .transforms import CoordinateTransformer


def detect_aruco_corners(
    frame: np.ndarray, aruco_dict_id: int = cv2.aruco.DICT_4X4_50
) -> np.ndarray | None:
    """检测 ArUco 标记，返回四个角点像素坐标"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None or len(ids) == 0:
        return None

    # 返回第一个检测到的标记的四个角点
    return corners[0][0]  # shape (4, 2)


def calibrate_with_clicks(frame: np.ndarray, board_size: int) -> np.ndarray:
    """手动点击棋盘四个角交叉点进行标定

    点击顺序: 左上(0,0) → 右上(0,n) → 右下(n,n) → 左下(n,0)
    """
    clicked_points = []
    display = frame.copy()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
            clicked_points.append((x, y))
            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
            labels = ["左上(0,0)", "右上(0,n)", "右下(n,n)", "左下(n,0)"]
            if len(clicked_points) <= 4:
                idx = len(clicked_points) - 1
                cv2.putText(
                    display, labels[idx], (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
                )
            cv2.imshow("Calibration", display)

    cv2.imshow("Calibration", display)
    cv2.setMouseCallback("Calibration", on_click)
    print("请依次点击棋盘四个角的交叉点:")
    print("  1. 左上 (0,0)")
    print("  2. 右上 (0,n)")
    print("  3. 右下 (n,n)")
    print("  4. 左下 (n,0)")

    while len(clicked_points) < 4:
        key = cv2.waitKey(100)
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            raise RuntimeError("用户取消标定")

    cv2.destroyAllWindows()
    return np.array(clicked_points, dtype=np.float32)


def compute_homography(
    pixel_corners: np.ndarray, board_size: int
) -> np.ndarray:
    """从四个角点计算像素→棋盘坐标的单应性矩阵

    Args:
        pixel_corners: 四个角点像素坐标 (4, 2)，顺序: 左上→右上→右下→左下
        board_size: 棋盘大小 (如 9)

    Returns:
        3×3 单应性矩阵
    """
    n = board_size - 1
    # 棋盘坐标 (col, row) 对应 (x, y)
    board_corners = np.array([
        [0, 0],      # 左上
        [n, 0],      # 右上
        [n, n],      # 右下
        [0, n],      # 左下
    ], dtype=np.float32)

    H, _ = cv2.findHomography(pixel_corners, board_corners)
    return H


def main():
    cfg = GomokuRobotConfig()

    print(f"打开相机 {cfg.camera.index} ...")
    cap = cv2.VideoCapture(cfg.camera.index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)

    if not cap.isOpened():
        print("无法打开相机!")
        return

    print("按 'c' 进行手动点击标定")
    print("按 'a' 尝试 ArUco 自动标定")
    print("按 'q' 退出")

    transformer = CoordinateTransformer(cfg.board, cfg.tray)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Board Calibration", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("c"):
            try:
                corners = calibrate_with_clicks(frame, cfg.board.size)
                H = compute_homography(corners, cfg.board.size)
                transformer.H_pixel_to_board = H
                transformer.save()
                print(f"标定完成! 单应性矩阵已保存到 {CALIBRATION_DIR}/transforms.json")

                # 验证: 在图像上绘制网格
                _draw_grid_overlay(frame, transformer, cfg.board.size)
            except RuntimeError as e:
                print(f"标定失败: {e}")
        elif key == ord("a"):
            corners = detect_aruco_corners(frame, cfg.aruco.dictionary)
            if corners is not None:
                H = compute_homography(corners, cfg.board.size)
                transformer.H_pixel_to_board = H
                transformer.save()
                print("ArUco 自动标定完成!")
            else:
                print("未检测到 ArUco 标记，请使用手动标定 ('c')")

    cap.release()
    cv2.destroyAllWindows()


def _draw_grid_overlay(
    frame: np.ndarray, transformer: CoordinateTransformer, board_size: int
):
    """在图像上绘制标定后的棋盘网格用于验证"""
    overlay = frame.copy()
    for r in range(board_size):
        for c in range(board_size):
            u, v = transformer.board_to_pixel(r, c)
            cv2.circle(overlay, (int(u), int(v)), 3, (0, 255, 0), -1)
            if c < board_size - 1:
                u2, v2 = transformer.board_to_pixel(r, c + 1)
                cv2.line(overlay, (int(u), int(v)), (int(u2), int(v2)), (0, 255, 0), 1)
            if r < board_size - 1:
                u2, v2 = transformer.board_to_pixel(r + 1, c)
                cv2.line(overlay, (int(u), int(v)), (int(u2), int(v2)), (0, 255, 0), 1)

    cv2.imshow("Calibration Verification", overlay)
    cv2.waitKey(0)
    cv2.destroyWindow("Calibration Verification")


if __name__ == "__main__":
    main()
