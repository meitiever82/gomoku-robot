"""
操控模块 — ACT 策略推理部署

加载训练好的 ACT 策略，执行 pick-and-place 动作。
"""

import numpy as np

try:
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.robots.so_follower import SO100Follower
    from lerobot.robots.so_follower.config_so_follower import SO100FollowerConfig
    from lerobot.cameras.configs import OpenCVCameraConfig
    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False


class GomokuArm:
    """五子棋机械臂控制器

    封装 LeRobot 的 SO-100 控制和 ACT 策略推理。
    """

    def __init__(
        self,
        policy_path: str,
        follower_port: str = "/dev/ttyACM0",
        camera_index: int = 0,
        board_size: int = 9,
        device: str = "cuda",
    ):
        if not HAS_LEROBOT:
            raise ImportError("需要安装 lerobot: pip install 'gomoku-robot[robot]'")

        self.board_size = board_size
        self.device = device

        # 加载策略
        self.policy = ACTPolicy.from_pretrained(policy_path)
        self.policy.to(device)
        self.policy.eval()

        # 配置机器人
        self.robot_cfg = SO100FollowerConfig(
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
        self.robot: SO100Follower | None = None

    def connect(self):
        self.robot = SO100Follower(self.robot_cfg)
        self.robot.connect()

    def disconnect(self):
        if self.robot is not None:
            self.robot.disconnect()
            self.robot = None

    def go_home(self):
        """回到安全位置（不遮挡相机）

        TODO: 需要根据实际标定确定 home 位置的关节角
        """
        if self.robot is None:
            raise RuntimeError("机器人未连接")
        # 预设的安全位置 (需要根据实际标定调整)
        home_joints = {
            "shoulder_pan.pos": 180.0,
            "shoulder_lift.pos": 180.0,
            "elbow_flex.pos": 180.0,
            "wrist_flex.pos": 180.0,
            "wrist_roll.pos": 180.0,
            "gripper.pos": 0.0,  # 夹爪打开
        }
        self.robot.send_action(home_joints)

    def pick_and_place(
        self,
        target_row: int,
        target_col: int,
        max_steps: int = 150,
    ):
        """执行 pick-and-place 动作

        Args:
            target_row: 目标棋盘行
            target_col: 目标棋盘列
            max_steps: 最大推理步数
        """
        if self.robot is None:
            raise RuntimeError("机器人未连接")

        # 归一化目标位置
        max_idx = self.board_size - 1
        target_norm = np.array([
            2.0 * target_col / max_idx - 1.0,
            2.0 * target_row / max_idx - 1.0,
        ], dtype=np.float32)

        # 重置策略的 action queue
        self.policy.reset()

        for step in range(max_steps):
            # 获取观测
            obs = self.robot.get_observation()

            # 注入目标位置
            obs["observation.environment_state"] = torch.tensor(
                target_norm, dtype=torch.float32
            ).unsqueeze(0).to(self.device)

            # 策略推理
            with torch.no_grad():
                action = self.policy.select_action(obs)

            # 执行动作
            self.robot.send_action(action)

        print(f"  pick-and-place 完成: ({target_row}, {target_col})")
