"""
五子棋机器人主循环

Usage:
  gomoku-play              # 完整模式 (需要硬件)
  python -m gomoku_robot.main --no-robot  # 纯软件模式 (终端对弈)
"""

import argparse
import numpy as np

from .config import GomokuRobotConfig
from .engine.gomoku_engine import GomokuEngine, BLACK, WHITE, EMPTY
from .engine.ai_player import GomokuAI


def play_terminal(board_size: int = 9, ai_depth: int = 4):
    """终端纯软件对弈模式 (不需要硬件)"""
    engine = GomokuEngine(board_size)
    ai = GomokuAI(board_size, max_depth=ai_depth)

    print("=" * 40)
    print("  五子棋机器人 — 终端模式")
    print(f"  棋盘: {board_size}×{board_size}")
    print("  你执白 (○)，AI 执黑 (●)")
    print("  输入格式: row col (如: 4 4)")
    print("  输入 q 退出, u 悔棋")
    print("=" * 40)

    current_color = BLACK  # 黑先

    while True:
        engine.print_board()

        state = engine.get_game_state()
        if state != "playing":
            if state == "black_wins":
                print("AI (●) 获胜!")
            elif state == "white_wins":
                print("你 (○) 获胜!")
            else:
                print("平局!")
            break

        if current_color == BLACK:
            # AI 回合
            print("AI 思考中...")
            move = ai.get_best_move(engine.board.copy(), BLACK)
            engine.place(move[0], move[1], BLACK)
            print(f"AI 落子: ({move[0]}, {move[1]})")
            current_color = WHITE
        else:
            # 人类回合
            while True:
                raw = input("你的落子 (row col): ").strip()
                if raw.lower() == "q":
                    print("退出游戏")
                    return
                if raw.lower() == "u":
                    # 悔两步 (自己和AI各一步)
                    engine.undo()
                    engine.undo()
                    engine.print_board()
                    continue
                try:
                    parts = raw.split()
                    r, c = int(parts[0]), int(parts[1])
                    if engine.place(r, c, WHITE):
                        current_color = BLACK
                        break
                    else:
                        print("无效位置，请重试")
                except (ValueError, IndexError):
                    print("格式错误，请输入: row col")


def play_with_robot():
    """完整硬件对弈模式"""
    from .calibration.transforms import CoordinateTransformer
    from .vision.board_detector import BoardDetector

    cfg = GomokuRobotConfig()
    engine = GomokuEngine(cfg.board.size)
    ai = GomokuAI(cfg.board.size, max_depth=4)

    # 加载标定
    transformer = CoordinateTransformer(cfg.board, cfg.tray)
    transformer.load()

    # 视觉
    detector = BoardDetector(cfg, transformer)
    detector.connect()

    piece_counter = 0  # 托盘棋子计数

    print("=" * 40)
    print("  五子棋机器人 — 硬件模式")
    print("  AI 执黑 (●)，你执白 (○)")
    print("=" * 40)

    current_color = BLACK

    try:
        while True:
            # 检测棋盘
            board_state = detector.detect_board()
            engine.board = board_state.copy()

            state = engine.get_game_state()
            if state != "playing":
                engine.print_board()
                if state == "black_wins":
                    print("AI (●) 获胜!")
                elif state == "white_wins":
                    print("你 (○) 获胜!")
                else:
                    print("平局!")
                break

            if current_color == BLACK:
                # AI 回合
                engine.print_board()
                print("AI 思考中...")
                move = ai.get_best_move(board_state.copy(), BLACK)
                print(f"AI 决定落子: ({move[0]}, {move[1]})")

                # 计算物理坐标
                pick_pos = transformer.tray_slot_to_robot(piece_counter)
                place_pos = transformer.board_to_robot(move[0], move[1])

                print(f"  取子位置: {pick_pos}")
                print(f"  落子位置: {place_pos}")

                # TODO: 调用 manipulation 模块执行 pick-and-place
                # arm.pick_and_place(pick_pos, place_pos)

                input("  [模拟] 请手动放置黑子到指定位置，然后按 Enter...")
                piece_counter += 1
                current_color = WHITE
            else:
                # 人类回合
                input("轮到你了! 放置白子后按 Enter...")
                current_color = BLACK

    finally:
        detector.disconnect()


def main():
    parser = argparse.ArgumentParser(description="五子棋机器人")
    parser.add_argument(
        "--no-robot", action="store_true", help="纯终端模式，不需要硬件"
    )
    parser.add_argument(
        "--board-size", type=int, default=9, help="棋盘大小 (默认 9)"
    )
    parser.add_argument(
        "--depth", type=int, default=4, help="AI 搜索深度 (默认 4)"
    )
    args = parser.parse_args()

    if args.no_robot:
        play_terminal(args.board_size, args.depth)
    else:
        play_with_robot()


if __name__ == "__main__":
    main()
