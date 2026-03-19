#!/usr/bin/env python3
"""LeIsaac SO-101 PickOrange 仿真验证脚本.

用法:
    source .venv-sim/bin/activate
    python scripts/test_sim.py

国内用户:
    HF_ENDPOINT=https://hf-mirror.com python scripts/test_sim.py

注意: 需要 ≥16GB GPU VRAM.
"""

import sys
import time

import torch


def main():
    print("=" * 50)
    print("  LeIsaac SO-101 PickOrange 仿真验证")
    print("=" * 50)

    # GPU 信息
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"GPU: {gpu_name} ({gpu_mem:.0f}GB)")
    else:
        print("错误: CUDA 不可用")
        sys.exit(1)

    print()
    print("加载 LeIsaac 环境 (首次会下载 ~250 个资产文件)...")
    t0 = time.time()

    from lerobot.envs.factory import make_env

    envs_dict = make_env(
        "LightwheelAI/leisaac_env:envs/so101_pick_orange.py",
        n_envs=1,
        trust_remote_code=True,
    )
    suite_name = next(iter(envs_dict))
    env = envs_dict[suite_name][0].envs[0].unwrapped

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
        print(f"  Step {i + 1}/{n_steps}: {dt:.2f}s (reward={reward:.4f})")
        sys.stdout.flush()
        if terminated or truncated:
            obs, info = env.reset()
            print(f"  (环境重置)")

    env.close()

    print()
    print("=" * 50)
    print("  LeIsaac SO-101 仿真验证通过!")
    print("=" * 50)


if __name__ == "__main__":
    main()
