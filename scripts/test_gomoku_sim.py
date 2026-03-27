#!/usr/bin/env python3
"""Gomoku PlaceGomokuStone 仿真验证脚本.

验证五子棋 pick-and-place 仿真环境可正常运行.

前置:
    1. 运行 python scripts/create_gomoku_scene.py 生成 USD 场景
    2. source .venv-sim/bin/activate

用法:
    python scripts/test_gomoku_sim.py

国内用户:
    HF_ENDPOINT=https://hf-mirror.com python scripts/test_gomoku_sim.py

8GB GPU 自动启用低显存模式.
"""

import os
import sys
import time

import torch


def main():
    print("=" * 50)
    print("  Gomoku PlaceGomokuStone 仿真验证")
    print("=" * 50)

    # GPU check
    if not torch.cuda.is_available():
        print("错误: CUDA 不可用")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} ({gpu_mem_gb:.0f}GB)")

    low_vram = gpu_mem_gb < 16
    if low_vram:
        print("  VRAM < 16GB, 启用低显存模式 (headless + 160x120 + 关闭RTX后处理)")

    # Download LeIsaac assets (SO-101 robot model etc.)
    print()
    print("下载资产文件 (首次可能较慢)...")
    from pathlib import Path

    from huggingface_hub import snapshot_download

    snapshot_dir = snapshot_download(repo_id="LightwheelAI/leisaac_env", revision=None, cache_dir=None)

    # Set LEISAAC_ASSETS_ROOT to local leisaac/assets (which has our gomoku scene)
    # but we also need the robot USD from the downloaded assets.
    # Strategy: use the local assets dir, and copy/symlink robot from downloaded.
    project_root = Path(__file__).resolve().parents[2]
    local_assets = project_root / "leisaac" / "assets"

    # Ensure robot model is available locally
    downloaded_robots = Path(snapshot_dir) / "assets" / "robots"
    local_robots = local_assets / "robots"
    if downloaded_robots.exists():
        for usd_file in downloaded_robots.glob("*.usd"):
            target = local_robots / usd_file.name
            if not target.exists():
                os.symlink(usd_file, target)
                print(f"  已链接: {usd_file.name}")

    os.environ["LEISAAC_ASSETS_ROOT"] = str(local_assets)
    print(f"  LEISAAC_ASSETS_ROOT = {local_assets}")

    # Check that the gomoku scene USD exists
    gomoku_usd = local_assets / "scenes" / "gomoku_table" / "scene.usd"
    if not gomoku_usd.exists():
        print(f"\n错误: 五子棋场景 USD 不存在: {gomoku_usd}")
        print("请先运行: python scripts/create_gomoku_scene.py")
        sys.exit(1)
    print(f"  五子棋场景: {gomoku_usd}")

    # Launch Isaac Sim
    print("\n启动 Isaac Sim...")
    t0 = time.time()

    from isaaclab.app import AppLauncher

    launcher_cfg = {"enable_cameras": True}
    if low_vram:
        launcher_cfg["headless"] = True
    _ = AppLauncher(launcher_cfg)

    # Low-VRAM: disable RTX post-processing
    if low_vram:
        import carb.settings

        s = carb.settings.get_settings()
        s.set("/rtx/post/dlss/enabled", False)
        s.set("/rtx/post/aa/op", 0)
        s.set("/rtx/directLighting/sampledLighting/enabled", False)
        s.set("/rtx/ambientOcclusion/enabled", False)
        s.set("/rtx/reflections/enabled", False)
        s.set("/rtx/translucency/enabled", False)
        s.set("/rtx/indirectDiffuse/enabled", False)
        print("  已禁用 DLSS 和 RTX 后处理效果")

    import gymnasium as gym
    import leisaac.tasks.place_gomoku_stone  # noqa: F401 — register gym env
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg("LeIsaac-SO101-PlaceGomokuStone-v0", device="cuda:0", num_envs=1)

    # Low-VRAM: reduce camera resolution
    if low_vram:
        env_cfg.scene.front.width = 160
        env_cfg.scene.front.height = 120

    env_cfg.use_teleop_device("so101leader")
    env_cfg.recorders = None

    env = gym.make("LeIsaac-SO101-PlaceGomokuStone-v0", cfg=env_cfg)

    t_load = time.time() - t0
    print(f"环境加载完成 ({t_load:.1f}s)")

    # Reset
    print("\n执行 env.reset()...")
    t0 = time.time()
    obs, info = env.reset()
    t_reset = time.time() - t0
    print(f"Reset 完成 ({t_reset:.1f}s)")

    # Print observation structure
    print("\n观测空间:")
    if isinstance(obs, dict):
        for key, val in obs.items():
            if isinstance(val, dict):
                for k2, v2 in val.items():
                    shape = v2.shape if hasattr(v2, "shape") else type(v2)
                    print(f"  {key}.{k2}: {shape}")
            elif hasattr(val, "shape"):
                print(f"  {key}: {val.shape}")
    else:
        print(f"  type: {type(obs)}, shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")

    # Step
    n_steps = 10
    print(f"\n执行 {n_steps} 步随机动作...")
    for i in range(n_steps):
        t0 = time.time()
        action = torch.tensor(env.action_space.sample())
        obs, reward, terminated, truncated, info = env.step(action)
        dt = time.time() - t0
        print(f"  Step {i + 1}/{n_steps}: {dt:.2f}s (reward={reward.item():.4f})")
        sys.stdout.flush()
        if terminated or truncated:
            obs, info = env.reset()
            print("  (环境重置)")

    env.close()

    print()
    print("=" * 50)
    print("  Gomoku PlaceGomokuStone 仿真验证通过!")
    print("=" * 50)


if __name__ == "__main__":
    main()
