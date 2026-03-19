"""
ACT 策略训练脚本

Usage:
  python -m gomoku_robot.manipulation.train_policy \
    --dataset-repo-id <user>/gomoku-pickplace-v1 \
    --output-dir outputs/gomoku-act \
    --steps 50000
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="训练五子棋 pick-and-place ACT 策略")
    parser.add_argument("--dataset-repo-id", required=True, help="LeRobot 数据集 ID")
    parser.add_argument("--output-dir", default="outputs/gomoku-act", help="输出目录")
    parser.add_argument("--steps", type=int, default=50000, help="训练步数")
    parser.add_argument("--batch-size", type=int, default=8, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument("--chunk-size", type=int, default=50, help="ACT chunk size")
    parser.add_argument("--device", default="cuda", help="训练设备")
    args = parser.parse_args()

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
    except ImportError:
        print("需要安装 lerobot: pip install 'gomoku-robot[robot]'")
        return

    print(f"加载数据集: {args.dataset_repo_id}")
    metadata = LeRobotDatasetMetadata(args.dataset_repo_id)
    print(f"  Episodes: {metadata.total_episodes}")
    print(f"  Frames: {metadata.total_frames}")

    # delta_timestamps: 控制输入输出的时间窗口
    delta_timestamps = {
        "action": [i / 30 for i in range(args.chunk_size)],
        "observation.images.top": [0.0],
        "observation.state": [0.0],
        "observation.environment_state": [0.0],
    }

    dataset = LeRobotDataset(
        args.dataset_repo_id,
        delta_timestamps=delta_timestamps,
    )

    # 配置 ACT 策略
    act_cfg = ACTConfig(
        chunk_size=args.chunk_size,
        n_action_steps=25,
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        dim_model=512,
        n_heads=8,
        n_encoder_layers=4,
        n_decoder_layers=1,
        use_vae=True,
        latent_dim=32,
    )

    print(f"\n训练配置:")
    print(f"  Steps: {args.steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  Device: {args.device}")
    print(f"  Output: {args.output_dir}")

    # TODO: 完整训练循环
    # 实际训练推荐直接使用 lerobot-train CLI:
    #
    # lerobot-train \
    #   --policy.type=act \
    #   --dataset.repo_id=<user>/gomoku-pickplace-v1 \
    #   --training.steps=50000 \
    #   --training.batch_size=8 \
    #   --training.lr=1e-5 \
    #   --policy.chunk_size=50 \
    #   --device=cuda
    #
    print("\n提示: 推荐直接使用 lerobot-train CLI 进行训练:")
    print(f"  lerobot-train \\")
    print(f"    --policy.type=act \\")
    print(f"    --dataset.repo_id={args.dataset_repo_id} \\")
    print(f"    --training.steps={args.steps} \\")
    print(f"    --training.batch_size={args.batch_size} \\")
    print(f"    --training.lr={args.lr} \\")
    print(f"    --device={args.device}")


if __name__ == "__main__":
    main()
