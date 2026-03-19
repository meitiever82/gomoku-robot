#!/usr/bin/env bash
# 五子棋机器人 — 仿真环境一键搭建脚本
# 用法: bash scripts/setup_sim.sh
# 国内用户: HF_ENDPOINT=https://hf-mirror.com bash scripts/setup_sim.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv-sim"

echo "============================================"
echo "  五子棋机器人 - 仿真环境搭建"
echo "============================================"
echo "项目目录: $PROJECT_DIR"
echo ""

# ---- 检查 NVIDIA GPU ----
echo "[1/10] 检查 GPU..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "错误: 未找到 nvidia-smi，请确认 NVIDIA 驱动已安装"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ---- 安装 uv ----
echo "[2/10] 检查 uv..."
if ! command -v uv &>/dev/null; then
    echo "安装 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv $(uv --version)"
echo ""

# ---- 创建虚拟环境 ----
echo "[3/10] 创建 Python 3.11 虚拟环境..."
if [ -d "$VENV_DIR" ]; then
    echo "已存在 $VENV_DIR，跳过创建"
else
    uv venv --python 3.11 "$VENV_DIR"
fi
PY="$VENV_DIR/bin/python"
echo "$($PY --version)"
echo ""

# ---- 安装 PyTorch ----
echo "[4/10] 安装 PyTorch 2.7.0 + CUDA 12.8..."
uv pip install --python "$PY" \
    torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128
echo ""

# ---- 安装 Isaac Sim ----
echo "[5/10] 安装 Isaac Sim 5.1.0..."
uv pip install --python "$PY" \
    isaacsim==5.1.0 \
    --extra-index-url https://pypi.nvidia.com
echo ""

# ---- 安装 flatdict (需要特殊处理) ----
echo "[6/10] 安装 flatdict (workaround)..."
uv pip install --python "$PY" setuptools
uv pip install --python "$PY" --no-build-isolation flatdict==4.0.1
echo ""

# ---- 安装 LeIsaac ----
echo "[7/10] 安装 LeIsaac + IsaacLab (需要下载 git 仓库，可能较慢)..."
uv pip install --python "$PY" \
    'leisaac[isaaclab] @ git+https://github.com/LightwheelAI/leisaac.git#subdirectory=source/leisaac' \
    --extra-index-url https://pypi.nvidia.com
echo ""

# ---- 安装 LeRobot + numpy 修复 ----
echo "[8/10] 安装 LeRobot 0.4.1 + 固定 numpy..."
uv pip install --python "$PY" lerobot==0.4.1
uv pip install --python "$PY" numpy==1.26.0
echo ""

# ---- 安装本项目 ----
echo "[9/10] 安装 gomoku-robot + 额外依赖..."
uv pip install --python "$PY" -e "$PROJECT_DIR"
uv pip install --python "$PY" numpy==1.26.0 opencv-contrib-python pytest
echo ""

# ---- 验证 ----
echo "[10/10] 验证安装..."
echo ""

# 接受 EULA
echo "  接受 Isaac Sim EULA..."
echo "Yes" | "$PY" -c "import isaacsim; print('  ✓ Isaac Sim OK')" 2>/dev/null

# PyTorch CUDA
"$PY" -c "
import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
print(f'  ✓ PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {gpu}')
"

# LeRobot
"$PY" -c "from lerobot.envs.factory import make_env; print('  ✓ LeRobot EnvHub OK')"

# gomoku-robot
"$PY" -c "import gomoku_robot; print('  ✓ gomoku-robot OK')"

echo ""
echo "============================================"
echo "  环境搭建完成!"
echo "============================================"
echo ""
echo "激活环境:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "运行仿真验证 (需要 ≥16GB VRAM):"
echo "  python scripts/test_sim.py"
echo ""
if [ -n "${HF_ENDPOINT:-}" ]; then
    echo "HF 镜像: $HF_ENDPOINT"
    echo "运行仿真前请确保设置了 HF_ENDPOINT 环境变量"
    echo ""
fi
