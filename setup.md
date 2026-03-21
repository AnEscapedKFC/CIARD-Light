# CIARD 环境与运行说明

> 如果你使用的是 **RTX 5070 Ti / Blackwell**，不要继续使用本文档里的
> `environment.yml`。请改用根目录下的 `setup_rtx5070ti.md`，因为当前
> 这份说明针对的是旧版 `PyTorch 2.3.1 + CUDA 11.8` 环境。

## 1. 前置条件

- 操作系统：Windows + PowerShell
- GPU：NVIDIA RTX 3060 Laptop GPU
- 驱动：安装较新的 NVIDIA 显卡驱动
- 环境管理：Miniconda 或 Anaconda

这个仓库通常不需要你额外安装完整的 CUDA Toolkit。
`environment.yml` 中使用了 `pytorch-cuda=11.8`，PyTorch 运行所需的 CUDA
运行时会随环境一起安装。

## 2. 创建环境

在项目根目录执行：

```powershell
conda env create -f environment.yml
conda activate ciard
```

## 3. 环境创建后的验证步骤

先验证基础依赖是否可以正常导入：

```powershell
python -c "import torch, torchvision, torchattacks, numpy, loguru; print('base imports ok')"
```

再验证 GPU 是否可用：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

预期结果：

- 第一条命令输出 `base imports ok`
- `torch.cuda.is_available()` 应为 `True`
- 显卡名称应显示为 `RTX 3060 Laptop GPU` 或同类名称

如果这里 GPU 不可用，优先检查 NVIDIA 驱动是否安装正常，再重新激活环境测试。

## 4. 安装 AutoAttack

`attack_eval.py` 里使用了 `AutoAttack`，但 `autoattack` 并没有发布到 PyPI，
所以需要在环境创建完成后手动安装。

推荐方法：

```powershell
pip install git+https://github.com/fra31/auto-attack.git
```

备用方法：

```powershell
git clone https://github.com/fra31/auto-attack.git _deps\auto-attack
Copy-Item -Recurse _deps\auto-attack\autoattack .\autoattack
```

如果使用备用方法，项目根目录最终应包含：

```text
CIARD/
  attack_eval.py
  autoattack/
```

安装完成后可以验证：

```powershell
python -c "from autoattack import AutoAttack; print('autoattack ok')"
```

## 5. 准备教师模型

先创建所需目录：

```powershell
New-Item -ItemType Directory -Force models, models\nat_teacher_checkpoint | Out-Null
```

将鲁棒教师模型放到：

```text
models/model_cifar_wrn.pt
```

这个路径在 `CIARD.py` 和 `attack_eval.py` 中都是硬编码的。

将自然教师模型放到：

```text
models/nat_teacher_checkpoint/cifar10_resnnet56.pth
```

这个路径在 `CIARD.py` 中是硬编码的。注意文件名里确实是 `resnnet56`，
因为源码里就是这样写的。

## 6. 数据集

代码使用的是 `torchvision.datasets.CIFAR10(..., download=True)`，因此
CIFAR-10 会自动下载到：

```text
data/
```

你不需要手动准备数据集。

## 7. 开始训练

在项目根目录执行：

```powershell
python CIARD.py
```

`CIARD.py` 当前默认配置为：

- 学生模型：`mobilenet_v2()`
- 鲁棒教师：`wideresnet()`
- 自然教师：`cifar10_resnet56()`
- 训练轮数：`300`
- `batch_size`：`128`
- 可见 GPU：`CUDA_VISIBLE_DEVICES=0`

训练输出会写入：

```text
model/Cifar10_MobileNetV2/
```

常见输出文件包括：

- `student_best.pth`
- `student_latest.pth`
- `teacher_best.pth`
- `teacher_latest.pth`

如果在 RTX 3060 Laptop GPU 上遇到 CUDA 显存不足，可以先把
`CIARD.py` 里的 `batch_size` 从 `128` 调小到 `64`。

## 8. 运行评测

在运行评测前，先修改 `attack_eval.py` 里的这一行：

```python
path = ""
```

替换成你要评测的学生模型路径，例如：

```python
path = "model/Cifar10_MobileNetV2/student_best.pth"
```

然后执行：

```powershell
python attack_eval.py
```

该评测脚本会运行：

- AutoAttack
- PGD
- FGSM
- CW Linf
- Square Attack
- 使用 `models/model_cifar_wrn.pt` 的黑盒攻击

## 9. 常用路径

- 训练脚本：`CIARD.py`
- 评测脚本：`attack_eval.py`
- 鲁棒教师模型：`models/model_cifar_wrn.pt`
- 自然教师模型：`models/nat_teacher_checkpoint/cifar10_resnnet56.pth`
- 训练输出目录：`model/Cifar10_MobileNetV2/`
- 数据集目录：`data/`

## 10. 快速开始

```powershell
conda env create -f environment.yml
conda activate ciard
python -c "import torch, torchvision, torchattacks, numpy, loguru; print('base imports ok')"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
pip install git+https://github.com/fra31/auto-attack.git
New-Item -ItemType Directory -Force models, models\nat_teacher_checkpoint | Out-Null
python CIARD.py
```
