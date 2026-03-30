# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Gomoku-robot is a 5-in-a-row (Gomoku) playing robot using a 6-DOF SO-100 robotic arm. The system combines traditional computer vision (ArUco markers + HSV color detection), alpha-beta search game AI, and learned manipulation policies (ACT via LeRobot). Documentation is in Chinese.

## Commands

```bash
# Install (editable, with robot hardware support)
pip install -e ".[robot]"

# Run tests
pytest gomoku_robot/tests/ -v

# Run a single test
pytest gomoku_robot/tests/test_engine.py::TestGomokuEngine::test_win_detection_horizontal -v

# Play in terminal (no hardware needed)
gomoku-play --no-robot --depth 4 --board-size 9

# Calibration
gomoku-calibrate-board
gomoku-calibrate-robot

# Simulation (requires Isaac Sim)
python scripts/play_gomoku_sim.py --depth 2 --gui
```

## Architecture

The system is a pipeline: **Camera → Vision → Game Engine → Robot Control → SO-100 Arm**

- **`gomoku_robot/engine/`** — Game logic and AI. `GomokuEngine` manages board state (0-indexed (row,col) tuples; EMPTY=0, BLACK=1/robot, WHITE=2/human). `GomokuAI` uses alpha-beta search (depth 4) with heuristic pattern scoring from `patterns.py`. Search is pruned to neighbors within radius-2 of existing pieces, top 15 candidates.

- **`gomoku_robot/vision/`** — Board detection via ArUco marker localization + HSV color thresholding at 81 grid intersections. No ML needed; runs <10ms/frame.

- **`gomoku_robot/calibration/`** — Three-stage coordinate transform chain: pixel (u,v) → board (row,col) via homography → robot (x,y,z) mm via rigid transform (SVD). Persisted to `calibration_data/transforms.json`.

- **`gomoku_robot/manipulation/`** — Robot control via ACT policy from LeRobot. `record_demos.py` collects leader-follower teleop data; `train_policy.py` scaffolds training (actual training uses `lerobot-train` CLI); `deploy_policy.py` runs inference with 50-frame action chunking.

- **`gomoku_robot/main.py`** — Game loop with two modes: terminal (`--no-robot`) and hardware. CLI entry points defined in `pyproject.toml`.

- **`gomoku_robot/config.py`** — Hierarchical dataclass configs (board, camera, ArUco, tray, vision, robot).

- **`scripts/`** — Isaac Sim simulation scripts.

## Dependencies

Core: numpy, opencv-contrib-python, torch. Optional `[robot]` extra: lerobot>=0.5.0. Simulation: isaaclab, leisaac.
