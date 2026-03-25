#!/usr/bin/env python3
"""LeIsaac SO-101 PickOrange 仿真验证脚本.

用法:
    source .venv-sim/bin/activate
    python scripts/test_sim.py

国内用户:
    HF_ENDPOINT=https://hf-mirror.com python scripts/test_sim.py

8GB GPU 自动启用低显存模式 (headless + 低分辨率 + 关闭RTX后处理).
>=16GB GPU 完整模式.
"""

import os
import sys
import time

import torch


def main():
    print("=" * 50)
    print("  LeIsaac SO-101 PickOrange 仿真验证")
    print("=" * 50)

    # GPU 信息
    if not torch.cuda.is_available():
        print("错误: CUDA 不可用")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} ({gpu_mem_gb:.0f}GB)")

    low_vram = gpu_mem_gb < 16
    if low_vram:
        print("  VRAM < 16GB, 启用低显存模式 (headless + 160x120 + 关闭DLSS)")

    print()
    print("下载资产文件 (首次可能较慢)...")
    from pathlib import Path

    from huggingface_hub import snapshot_download

    snapshot_dir = snapshot_download(repo_id="LightwheelAI/leisaac_env", revision=None, cache_dir=None)
    assets_root = Path(snapshot_dir) / "assets"
    os.environ["LEISAAC_ASSETS_ROOT"] = str(assets_root)

    print("启动 Isaac Sim...")
    t0 = time.time()

    from isaaclab.app import AppLauncher

    launcher_cfg = {"enable_cameras": True}
    if low_vram:
        launcher_cfg["headless"] = True
    _ = AppLauncher(launcher_cfg)

    # 低显存: 禁用 DLSS 和 RTX 后处理, 省显存
    if low_vram:
        import carb.settings

        s = carb.settings.get_settings()
        s.set("/rtx/post/dlss/enabled", False)
        s.set("/rtx/post/aa/op", 0)  # 关闭抗锯齿后处理
        s.set("/rtx/directLighting/sampledLighting/enabled", False)
        s.set("/rtx/ambientOcclusion/enabled", False)
        s.set("/rtx/reflections/enabled", False)
        s.set("/rtx/translucency/enabled", False)
        s.set("/rtx/indirectDiffuse/enabled", False)
        print("  已禁用 DLSS 和 RTX 后处理效果")

    import gymnasium as gym
    import leisaac.tasks.pick_orange  # noqa: F401 — 注册 gym 环境
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg("LeIsaac-SO101-PickOrange-v0", device="cuda:0", num_envs=1)

    # 低显存: 大幅降低摄像头分辨率 640x480 -> 160x120
    if low_vram:
        env_cfg.scene.wrist.width = 160
        env_cfg.scene.wrist.height = 120
        env_cfg.scene.front.width = 160
        env_cfg.scene.front.height = 120

    env_cfg.use_teleop_device("so101leader")
    env_cfg.recorders = None

    env = gym.make("LeIsaac-SO101-PickOrange-v0", cfg=env_cfg)

    t_load = time.time() - t0
    print(f"环境加载完成 ({t_load:.1f}s)")

    print()
    print("执行 env.reset()...")
    t0 = time.time()
    obs, info = env.reset()
    t_reset = time.time() - t0
    print(f"Reset 完成 ({t_reset:.1f}s)")

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
    print("  LeIsaac SO-101 仿真验证通过!")
    print("=" * 50)


if __name__ == "__main__":
    main()
