# CIARD 在 RTX 5070 Ti / Blackwell 上的环境与运行说明

这份说明专门面向 NVIDIA Blackwell 架构显卡，例如 RTX 5070 Ti。

不要继续复用仓库里旧的 `environment.yml` 环境，因为它固定了：

- `pytorch=2.3.1`
- `pytorch-cuda=11.8`
- `python=3.8`

这套组合适合较早一代 GPU，不适合作为 RTX 5070 Ti 的训练环境基线。

## 1. 前置条件

- 操作系统：Windows + PowerShell
- GPU：NVIDIA RTX 5070 Ti
- 驱动：建议使用 `570.65` 或更新版本
- 环境管理：Miniconda 或 Anaconda

这个项目不需要你单独安装完整 CUDA Toolkit。
PyTorch 官方 `cu128` 轮子已经自带运行时，足够支撑训练和评测。

先检查驱动：

```powershell
nvidia-smi
```

如果驱动版本低于 R570 分支，建议先升级驱动，再继续下面的步骤。

## 2. 创建新的 Blackwell 环境

不要在旧环境上直接覆盖升级，建议单独创建一个新环境。

可以直接使用仓库里的环境文件：

```powershell
conda env create -f environment-blackwell.yml
conda activate ciard-blackwell
python -m pip install --upgrade pip setuptools wheel
```

也可以手动创建：

```powershell
conda create -n ciard-blackwell python=3.10 git pip -y
conda activate ciard-blackwell
python -m pip install --upgrade pip setuptools wheel
```

如果这个环境已经被试验过，想彻底重建：

```powershell
conda deactivate
conda remove -n ciard-blackwell --all -y
conda env create -f environment-blackwell.yml
conda activate ciard-blackwell
python -m pip install --upgrade pip setuptools wheel
```

## 3. 安装 PyTorch 2.7.1 + CUDA 12.8

先安装支持 Blackwell 的官方 PyTorch 轮子：

```powershell
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

再安装项目本身依赖：

```powershell
python -m pip install -r requirements-blackwell.txt
```

`attack_eval.py` 还依赖 `AutoAttack`，需要单独安装：

```powershell
python -m pip install git+https://github.com/fra31/auto-attack.git
```

## 4. 验证 PyTorch 是否真正识别 RTX 5070 Ti

安装完成后，执行下面三条命令：

```powershell
python -c "import torch; print('torch =', torch.__version__); print('cuda =', torch.version.cuda); print('cuda available =', torch.cuda.is_available()); print('device =', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu'); print('capability =', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'n/a'); print('arch list =', torch.cuda.get_arch_list() if torch.cuda.is_available() else 'n/a')"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print('sm_120 in build =', 'sm_120' in torch.cuda.get_arch_list())"
python -c "import torch; x = torch.randn(2048, 2048, device='cuda'); y = x @ x; torch.cuda.synchronize(); print('matmul ok, mean =', float(y.abs().mean()))"
```

预期结果：

- `torch.__version__` 应为 `2.7.1`
- `torch.version.cuda` 应为 `12.8`
- `torch.cuda.is_available()` 应为 `True`
- 显卡名称应显示为 `NVIDIA GeForce RTX 5070 Ti`
- 计算能力应为 `(12, 0)`
- `sm_120 in build` 应输出 `True`

如果显示 `(12, 0)`，但 `sm_120 in build` 是 `False`，通常说明你仍在使用旧的 PyTorch 安装，或者当前环境被旧依赖污染了。

## 5. 基础依赖验证

```powershell
python -c "import torch, torchvision, torchattacks, numpy, loguru; print('base imports ok')"
python -c "from autoattack import AutoAttack; print('autoattack ok')"
```

## 6. 本次已做的代码适配

为了让项目更稳地跑在 `PyTorch 2.7.1 + cu128` 环境上，仓库里的脚本已经做了以下适配：

- `CIARD.py` 和 `attack_eval.py` 现在通过 `torch.cuda.is_available()` 自动选择设备，并统一改为 `.to(device)`，不再在脚本内部硬编码 `CUDA_VISIBLE_DEVICES=0`
- `CIARD.py`、`attack_eval.py` 对 checkpoint 加载增加了兼容处理，优先使用 `torch.load(..., weights_only=True)`，同时保留对旧版参数签名的回退
- `attack_eval.py` 改为使用 `torchattacks.Square(...)` 这个公开 API，而不是依赖包内部模块路径
- `attack_eval.py` 会在 `path` 为空时直接报错，避免误跑空路径
- `mtard_loss.py` 中对对抗样本和标签的处理改为跟随输入张量所在设备，避免设备写死
- `CIARD.py` 修复了 `loss3_weight` 在异常分支下可能未定义的问题，并为教师模型评测补上了 `torch.no_grad()`，减少无意义显存开销

如果你想手动指定使用哪张 GPU，请在启动脚本前，在 PowerShell 里设置：

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
python CIARD.py
```

不要再依赖脚本内部去改这个环境变量。

## 7. 准备教师模型与目录

先创建目录：

```powershell
New-Item -ItemType Directory -Force models, models\nat_teacher_checkpoint | Out-Null
```

教师模型路径保持不变：

- 鲁棒教师模型：`models/model_cifar_wrn.pt`
- 自然教师模型：`models/nat_teacher_checkpoint/cifar10_resnnet56.pth`

注意自然教师模型文件名里的 `resnnet56` 是源码里现有的硬编码名称，不要擅自改名。

## 8. 数据集

`CIARD.py` 和 `attack_eval.py` 都使用 `torchvision.datasets.CIFAR10(..., download=True)`。

因此 CIFAR-10 会自动下载到：

```text
data/
```

一般不需要你手动准备数据。

## 9. 开始训练

```powershell
python CIARD.py
```

训练输出目录：

```text
model/Cifar10_MobileNetV2/
```

如果在 5070 Ti 上仍然遇到显存压力，可以优先从 `CIARD.py` 中把 `batch_size` 继续调小。

## 10. 运行评测

先在 `attack_eval.py` 中设置学生模型路径：

```python
path = "model/Cifar10_MobileNetV2/student_best.pth"
```

再执行：

```powershell
python attack_eval.py
```

当前评测脚本会运行：

- AutoAttack
- PGD
- FGSM
- CW Linf
- Square Attack
- 基于 `models/model_cifar_wrn.pt` 的黑盒攻击

## 11. 一次性命令汇总

```powershell
nvidia-smi
conda env create -f environment-blackwell.yml
conda activate ciard-blackwell
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-blackwell.txt
python -m pip install git+https://github.com/fra31/auto-attack.git
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), 'sm_120' in torch.cuda.get_arch_list())"
python CIARD.py
```
