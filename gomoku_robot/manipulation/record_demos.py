"""
五子棋抓放示教录制脚本

基于 LeRobot 的 leader-follower 示教，录制 pick-and-place 数据。
每个 episode 录制: 从托盘拾取棋子 → 放到指定棋盘位置。

目标位置通过 observation.environment_state 注入数据集,
使 ACT 策略学会"放到哪里"。

Usage:
  python -m gomoku_robot.manipulation.record_demos \
    --follower-port /dev/ttyACM0 \
    --leader-port /dev/ttyACM1 \
    --num-episodes 40 \
    --repo-id <user>/gomoku-pickplace-v1
"""

import argparse
import time
import numpy as np

try:
    from lerobot.robots.so_follower import SO100Follower
    from lerobot.robots.so_follower.config_so_follower import SO100FollowerConfig
    from lerobot.teleoperators.so_leader import SO100Leader
    from lerobot.teleoperators.so_leader.config_so_leader import SO100LeaderConfig
    from lerobot.cameras.configs import OpenCVCameraConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False

from ..config import GomokuRobotConfig


def generate_target_positions(board_size: int = 9) -> list[tuple[int, int]]:
    """生成覆盖棋盘各区域的目标位置列表

    返回均匀分布在棋盘上的位置，用于数据采集。
    """
    positions = []

    # 中心
    c = board_size // 2
    positions.append((c, c))

    # 四个角
    positions.extend([(0, 0), (0, board_size-1), (board_size-1, 0), (board_size-1, board_size-1)])

    # 四条边中点
    positions.extend([(0, c), (c, 0), (board_size-1, c), (c, board_size-1)])

    # 内部均匀采样
    for r in range(1, board_size - 1, 2):
        for col in range(1, board_size - 1, 2):
            if (r, col) not in positions:
                positions.append((r, col))

    return positions


def normalize_target(row: int, col: int, board_size: int) -> np.ndarray:
    """将棋盘坐标归一化到 [-1, 1] 范围"""
    max_idx = board_size - 1
    x = 2.0 * col / max_idx - 1.0  # col → x
    y = 2.0 * row / max_idx - 1.0  # row → y
    return np.array([x, y], dtype=np.float32)


def record_episodes(
    follower_port: str,
    leader_port: str,
    camera_index: int,
    num_episodes: int,
    repo_id: str,
    board_size: int = 9,
):
    """录制示教数据"""
    if not HAS_LEROBOT:
        raise ImportError("需要安装 lerobot: pip install 'gomoku-robot[robot]'")

    # 生成目标位置
    targets = generate_target_positions(board_size)
    print(f"目标位置列表 ({len(targets)} 个): {targets[:10]}...")

    # 配置机器人
    follower_cfg = SO100FollowerConfig(
        port=follower_port,
        id="gomoku_follower",
        cameras={
            "top": OpenCVCameraConfig(
                index_or_path=camera_index,
                width=640,
                height=480,
                fps=30,
            ),
        },
    )

    leader_cfg = SO100LeaderConfig(
        port=leader_port,
        id="gomoku_leader",
    )

    follower = SO100Follower(follower_cfg)
    leader = SO100Leader(leader_cfg)

    follower.connect()
    leader.connect()

    print(f"\n开始录制 {num_episodes} 个 episodes")
    print("每个 episode:")
    print("  1. 显示目标位置")
    print("  2. 用 leader 臂示教: 从托盘拾取棋子 → 放到目标位置")
    print("  3. 按 Enter 结束当前 episode")
    print("  按 q 提前结束录制\n")

    recorded = 0
    for ep in range(num_episodes):
        target_idx = ep % len(targets)
        target_row, target_col = targets[target_idx]
        target_norm = normalize_target(target_row, target_col, board_size)

        print(f"\n--- Episode {ep + 1}/{num_episodes} ---")
        print(f"目标位置: ({target_row}, {target_col})")
        print(f"归一化坐标: ({target_norm[0]:.2f}, {target_norm[1]:.2f})")
        print("请将棋子放入托盘，准备好后按 Enter 开始录制...")

        cmd = input().strip()
        if cmd.lower() == "q":
            break

        print("录制中... 用 leader 臂示教抓放动作，完成后按 Enter")

        # TODO: 实际录制循环
        # 这里是框架代码，实际需要:
        # 1. 创建 LeRobotDataset writer
        # 2. 循环读取 leader 位置 → 发送给 follower
        # 3. 同时记录 observation (joints + camera + environment_state)
        # 4. environment_state = target_norm

        episode_data = []
        recording = True
        step = 0

        while recording:
            # 读取 leader 位置
            leader_obs = leader.get_observation()
            action = leader_obs  # leader 的位置就是 follower 的 action

            # 发送给 follower
            follower.send_action(action)

            # 读取 follower 观测
            obs = follower.get_observation()

            # 注入目标位置
            obs["observation.environment_state"] = target_norm

            episode_data.append({
                "observation": obs,
                "action": action,
                "step": step,
            })

            step += 1
            time.sleep(1.0 / 30)  # 30 fps

            # 检查是否按下 Enter (非阻塞)
            # 实际实现中需要更好的输入处理
            # 这里简化为固定时长或手动停止

        input("按 Enter 结束本 episode...")
        recorded += 1
        print(f"Episode {ep + 1} 完成, 共 {step} 帧")

    follower.disconnect()
    leader.disconnect()
    print(f"\n录制完成! 共 {recorded} 个 episodes")


def main():
    parser = argparse.ArgumentParser(description="五子棋抓放示教录制")
    parser.add_argument("--follower-port", default="/dev/ttyACM0")
    parser.add_argument("--leader-port", default="/dev/ttyACM1")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=40)
    parser.add_argument("--repo-id", default="gomoku-pickplace-v1")
    parser.add_argument("--board-size", type=int, default=9)
    args = parser.parse_args()

    record_episodes(
        follower_port=args.follower_port,
        leader_port=args.leader_port,
        camera_index=args.camera_index,
        num_episodes=args.num_episodes,
        repo_id=args.repo_id,
        board_size=args.board_size,
    )


if __name__ == "__main__":
    main()
