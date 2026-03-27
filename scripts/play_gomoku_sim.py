#!/usr/bin/env python3
"""在 Isaac Sim 仿真中运行双臂五子棋对弈.

两个 SO-101 机械臂面对面，一个执黑一个执白，AI vs AI。

用法:
    source .venv-sim/bin/activate
    python scripts/play_gomoku_sim.py [--depth 2] [--board-size 9] [--gui]

国内用户:
    HF_ENDPOINT=https://hf-mirror.com python scripts/play_gomoku_sim.py
"""

import argparse
import os
import sys
import time

# ---------------------------------------------------------------------------
# Isaac Sim bootstrap (MUST come before any omni/isaaclab imports)
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Gomoku AI vs AI - Dual Arm in Isaac Sim")
parser.add_argument("--depth", type=int, default=2, help="AI search depth (2=fast, 4=strong)")
parser.add_argument("--board-size", type=int, default=9, help="Board size")
parser.add_argument("--max-moves", type=int, default=50, help="Max moves before stopping")
parser.add_argument("--gui", action="store_true", help="Show GUI window (default: headless)")
AppLauncher.add_app_launcher_args(parser)
# Default to headless unless --gui is specified
if "--gui" not in sys.argv:
    if "--headless" not in sys.argv:
        sys.argv += ["--headless"]
if "--enable_cameras" not in sys.argv:
    sys.argv += ["--enable_cameras"]
args = parser.parse_args()

app_launcher = AppLauncher(vars(args))
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Imports (after SimulationApp)
# ---------------------------------------------------------------------------
import gymnasium as gym
import numpy as np
import torch

import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

# Set LEISAAC_ASSETS_ROOT BEFORE importing leisaac
from pathlib import Path
from huggingface_hub import snapshot_download

_snapshot_dir = snapshot_download(repo_id="LightwheelAI/leisaac_env", revision=None, cache_dir=None)
_project_root = Path(__file__).resolve().parents[2]
_local_assets = _project_root / "leisaac" / "assets"
_downloaded_robots = Path(_snapshot_dir) / "assets" / "robots"
_local_robots = _local_assets / "robots"
if _downloaded_robots.exists():
    for _usd in _downloaded_robots.glob("*.usd"):
        _target = _local_robots / _usd.name
        if not _target.exists():
            os.symlink(_usd, _target)
os.environ["LEISAAC_ASSETS_ROOT"] = str(_local_assets)

# Register our task
import leisaac.tasks.place_gomoku_stone  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# Game engine
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gomoku_robot.engine.gomoku_engine import BLACK, WHITE, GomokuEngine
from gomoku_robot.engine.ai_player import GomokuAI

# State machine
from leisaac.datagen.state_machine.place_gomoku_stone import PlaceGomokuStoneStateMachine

# ---------------------------------------------------------------------------
# Scene parameters (must match create_gomoku_scene.py)
# ---------------------------------------------------------------------------
GRID_SPACING = 0.030
BOARD_CENTER_X = 0.30
BOARD_CENTER_Y = 0.0
STONE_RADIUS = 0.012
STONE_HEIGHT = 0.010
TRAY_POS_X = 0.08
TRAY_POS_Y = -0.20
TRAY_Z = 0.02 / 2 + 0.005 / 2 + STONE_HEIGHT / 2

COLOR_BLACK = (0.08, 0.08, 0.08)
COLOR_WHITE = (0.92, 0.92, 0.92)


def grid_to_world(row: int, col: int) -> tuple[float, float]:
    """Convert board grid (row, col) to world XY."""
    half = 4.0
    x = BOARD_CENTER_X + (col - half) * GRID_SPACING
    y = BOARD_CENTER_Y + (row - half) * GRID_SPACING
    return x, y


def teleport_stone_to_tray(env, stone_name: str):
    """Teleport the active stone to the tray position."""
    stone = env.scene[stone_name]
    pos = torch.tensor([[TRAY_POS_X, TRAY_POS_Y, TRAY_Z]], device=env.device)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
    vel = torch.zeros(1, 6, device=env.device)
    stone.write_root_state_to_sim(torch.cat([pos, quat, vel], dim=-1))


def hide_stone(env, stone_name: str):
    """Move stone under the table (out of sight)."""
    stone = env.scene[stone_name]
    pos = torch.tensor([[0.0, 0.0, -0.5]], device=env.device)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
    vel = torch.zeros(1, 6, device=env.device)
    stone.write_root_state_to_sim(torch.cat([pos, quat, vel], dim=-1))


def freeze_stone_at_position(row: int, col: int, color: int, move_number: int):
    """Create a static stone on the board using USD API."""
    stage = omni.usd.get_context().get_stage()
    x, y = grid_to_world(row, col)
    board_top_z = 0.02 / 2 + 0.008 / 2
    z = board_top_z + STONE_HEIGHT / 2

    prim_path = f"/World/envs/env_0/Scene/World/PlacedStone_{move_number}"
    cyl = UsdGeom.Cylinder.Define(stage, prim_path)
    cyl.CreateRadiusAttr(STONE_RADIUS)
    cyl.CreateHeightAttr(STONE_HEIGHT)
    cyl.CreateAxisAttr("Z")
    cyl.AddTranslateOp().Set(Gf.Vec3d(x, y, z))

    prim = stage.GetPrimAtPath(prim_path)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    prim.GetAttribute("physics:kinematicEnabled").Set(True)
    UsdPhysics.CollisionAPI.Apply(prim)

    rgb = COLOR_BLACK if color == BLACK else COLOR_WHITE
    mat_path = f"{prim_path}/Material"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(prim).Bind(mat)


def main():
    board_size = args.board_size
    ai_depth = args.depth
    max_moves = args.max_moves
    render = args.gui

    print("=" * 50)
    print("  五子棋仿真对弈 — 双臂版 (AI vs AI)")
    print(f"  棋盘: {board_size}×{board_size}, AI depth: {ai_depth}")
    print("=" * 50)

    # --- Disable RTX post-processing (save VRAM) ---
    import carb.settings
    s = carb.settings.get_settings()
    s.set("/rtx/post/dlss/enabled", False)
    s.set("/rtx/post/aa/op", 0)
    s.set("/rtx/directLighting/sampledLighting/enabled", False)
    s.set("/rtx/ambientOcclusion/enabled", False)
    s.set("/rtx/reflections/enabled", False)
    s.set("/rtx/translucency/enabled", False)
    s.set("/rtx/indirectDiffuse/enabled", False)

    # --- Create bi-arm environment ---
    print("\n创建双臂仿真环境...")
    t0 = time.time()

    task_id = "LeIsaac-SO101-PlaceGomokuStone-BiArm-v0"
    env_cfg = parse_env_cfg(task_id, device="cuda:0", num_envs=1)
    env_cfg.scene.top.width = 160
    env_cfg.scene.top.height = 120
    env_cfg.use_teleop_device("bi_so101_state_machine")
    env_cfg.recorders = None

    env = gym.make(task_id, cfg=env_cfg).unwrapped
    env.reset()

    # Set viewport camera
    if render:
        from isaaclab.sim import SimulationContext
        sim = SimulationContext.instance()
        sim.set_camera_view(eye=(0.30, -0.65, 0.55), target=(0.30, 0.0, 0.02))

    print(f"环境就绪 ({time.time() - t0:.1f}s)")

    # --- Initialize game ---
    engine = GomokuEngine(board_size)
    ai_black = GomokuAI(board_size, max_depth=ai_depth)
    ai_white = GomokuAI(board_size, max_depth=ai_depth)

    # Two state machines: one per arm
    sm_black = PlaceGomokuStoneStateMachine(stone_name="Stone", robot_name="left_arm")
    sm_white = PlaceGomokuStoneStateMachine(stone_name="Stone", robot_name="right_arm")
    sm_black.setup(env)
    sm_white.setup(env)

    current_color = BLACK
    move_number = 0

    print("\n  左臂 (Left) = 黑棋 ●")
    print("  右臂 (Right) = 白棋 ○")
    print("\n开始对弈!\n")
    engine.print_board()

    game_start = time.time()

    # --- Determine action dimensions ---
    # bi_so101_state_machine: left_arm(7) + left_gripper(1) + right_arm(7) + right_gripper(1) = 16
    action_dim = env.action_space.shape[-1]

    # --- Main game loop ---
    while move_number < max_moves:
        # 1. AI computes move
        ai = ai_black if current_color == BLACK else ai_white
        t0 = time.time()
        move = ai.get_best_move(engine.board.copy(), current_color)
        ai_time = time.time() - t0
        row, col = move

        sm = sm_black if current_color == BLACK else sm_white
        arm_name = "左臂 ●" if current_color == BLACK else "右臂 ○"
        print(f"Move {move_number + 1}: {arm_name} → ({row}, {col})  [AI: {ai_time:.2f}s]", end="", flush=True)

        # 2. Teleport stone to tray
        teleport_stone_to_tray(env, "Stone")
        for _ in range(10):
            env.sim.step(render=render)
            env.scene.update(dt=env.physics_dt)

        # 3. Run state machine for the active arm
        sm.reset(target_row=row, target_col=col)
        t0 = time.time()
        step_count = 0
        while not sm.is_episode_done:
            sm.pre_step(env)
            arm_action = sm.get_action(env)  # 8D for active arm

            # Build full action: [left_arm(8), right_arm(8)]
            # Idle arm gets zero action (hold position)
            full_action = torch.zeros(1, action_dim, device=env.device)
            if current_color == BLACK:
                full_action[:, :8] = arm_action  # left arm active
            else:
                full_action[:, 8:] = arm_action  # right arm active

            env.step(full_action)
            sm.advance()
            step_count += 1

        sim_time = time.time() - t0
        print(f"  [Sim: {step_count} steps, {sim_time:.1f}s]", flush=True)

        # 4. Freeze stone at board position, hide active stone
        freeze_stone_at_position(row, col, current_color, move_number)
        hide_stone(env, "Stone")
        for _ in range(5):
            env.sim.step(render=render)
            env.scene.update(dt=env.physics_dt)

        # 5. Update game engine
        engine.place(row, col, current_color)
        move_number += 1

        # Print board
        color_sym = "BLACK ●" if current_color == BLACK else "WHITE ○"
        print(f"\n--- Move {move_number}: {color_sym} at ({row}, {col}) ---")
        engine.print_board()

        # 6. Check game state
        state = engine.get_game_state()
        if state != "playing":
            if state == "black_wins":
                print("\n结果: 黑棋 ● (左臂) 获胜!")
            elif state == "white_wins":
                print("\n结果: 白棋 ○ (右臂) 获胜!")
            else:
                print("\n结果: 平局!")
            break

        # 7. Switch color
        current_color = WHITE if current_color == BLACK else BLACK

    total_time = time.time() - game_start
    print(f"\n共 {move_number} 步, 总耗时 {total_time:.0f}s")

    if engine.get_game_state() == "playing":
        print(f"\n达到最大步数 {max_moves}, 游戏未结束")
        engine.print_board()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
