# LiteAVSE 环境配置说明（Windows）

## 一、已完成的配置

- **虚拟环境**：项目根目录下已创建 `.venv`
- **核心依赖**：已通过 `requirements_core.txt` 安装（torch、librosa、opencv 等）

## 二、激活虚拟环境

在项目根目录下，PowerShell 执行：

```powershell
.\.venv\Scripts\Activate.ps1
```

若提示无法执行脚本，先运行：

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 三、仍需完成的步骤

### 1. 安装 PESQ（评估指标，需 C++ 编译环境）

`pesq` 在 Windows 上需要 **Microsoft C++ 生成工具** 才能从源码编译。

**步骤：**

1. 安装 [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)（选择“使用 C++ 的桌面开发”工作负载）。
2. 安装完成后**重新打开终端**，激活 `.venv`，再执行：

   ```powershell
   pip install pesq
   ```

若仍失败，可尝试使用 Python 3.10 或 3.11 创建新的 venv（对预编译 wheel 支持更好）。

### 2. 安装 mamba-ssm（Mamba 模块）

本项目使用 Mamba 做语音增强 backbone。**Windows 上官方未提供预编译包**，通常需要：

**方案 A：从源码安装（需 CUDA + 正确环境变量）**

- 确保已安装 CUDA Toolkit，且 `nvcc` 可用。
- 确保已安装 PyTorch（带 CUDA 版本）：  
  https://pytorch.org/get-started/locally/ 选择 Windows + CUDA 后给出的 `pip` 命令。
- 克隆并安装：

  ```powershell
  git clone https://github.com/state-spaces/mamba.git
  cd mamba
  pip install -e . --no-build-isolation --no-cache-dir --no-binary=:all:
  ```

**方案 B：使用 WSL2 或 Linux**

在 WSL2/Linux 下按 README 执行 `pip install mamba-ssm` 或从源码安装，成功率更高。

**说明**：当前代码中 `models/mamba_block.py` 的 import 针对 mamba-ssm 1.2.x；若安装的是 2.x，需按文件内 TODO 修改 import 路径。

### 3. 可选：安装 causal-conv1d（mamba-ssm 2.x 依赖）

若使用 mamba-ssm 2.x：

```powershell
pip install causal-conv1d>=1.4.0
```

## 四、验证环境

激活 `.venv` 后：

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import torchaudio, librosa, yaml, cv2; print('核心库 OK')"
```

若已安装 pesq：

```powershell
python -c "from pesq import pesq; print('pesq OK')"
```

若已安装 mamba-ssm：

```powershell
python -c "from mamba_ssm.modules.mamba_simple import Mamba; print('mamba-ssm OK')"
```

## 五、运行训练（需数据与配置）

- 仅音频 baseline：  
  `python train.py --config recipes/SEMamba_advanced/SEMamba_advanced.yaml`
- 音视频 LiteAVSE：  
  `python train_lite.py --config recipes/LiteAVSE/LiteAVSE.yaml --exp_folder exp --exp_name LiteAVSE_v1`

数据需按 README 中“Datasets”部分准备，并运行 `data/` 下脚本生成 JSON 列表。

## 六、快速安装脚本

已提供 `scripts/setup_env.ps1`，在项目根目录执行：

```powershell
.\scripts\setup_env.ps1
```

会创建/使用 `.venv` 并安装 `requirements_core.txt`；PESQ 和 mamba-ssm 仍需按上文手动安装。
