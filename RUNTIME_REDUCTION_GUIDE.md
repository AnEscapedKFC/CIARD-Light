# CIARD 运行时间缩短说明

这份说明只描述“如何改”，不直接修改现有代码文件。目标是在尽量不改变实验结论的前提下，先把训练和评估时间降下来。

## 一、先说结论

如果你希望“尽量不影响结论”，优先按下面顺序做：

1. 保留完整测试集，只缩小训练集。
2. 减少训练轮次，但要同步修改学习率日程。
3. 降低中间评估频率。
4. 只在最终 checkpoint 上做最重的评估（例如 AutoAttack）。

不建议一开始就动下面这些关键设定：

- `epsilon = 8/255`
- 学生模型结构
- 两个 teacher checkpoint
- 随机种子

这些更直接决定结果可比性，最好先保持不变。

## 二、当前耗时主要来自哪里

训练脚本的主要耗时点在 [CIARD.py](F:\Softmax-Distillation\CIARD\CIARD.py)：

- 第 33 行：`epochs = 300`
- 第 63-67 行：使用完整 CIFAR-10 训练集和测试集
- 第 208-212 行：每个训练 batch 都会调用 `robust_inner_loss_push(..., perturb_steps=10)`
- 第 303-325 行：`epoch == 1`、每 10 个 epoch、以及 `epoch >= 250` 时都会做测试
- 第 325 行：测试阶段每个 batch 都会做 `attack_pgd(..., attack_iters=20)`

评估脚本的主要耗时点在 [attack_eval.py](F:\Softmax-Distillation\CIARD\attack_eval.py)：

- 第 101-102 行：默认直接跑 AutoAttack
- 第 126、143、204 行：多次 `PGD-20`
- 第 218 行：Square Attack
- 第 241 行：CW 攻击

## 三、推荐的低风险缩时方案

下面这组改法最适合“先跑通并看趋势，同时尽量保持结论不变”。

### 方案 A：保留完整测试集，只使用分层抽样后的训练子集

修改位置：

- [CIARD.py](F:\Softmax-Distillation\CIARD\CIARD.py) 第 63-64 行附近

思路：

- 不动 `testset`
- 只对 `trainset` 做类别均衡抽样
- 每个类别保留固定数量样本，例如每类 `2000` 张，总计 `20000` 张

建议：

- `20000` 张训练样本：通常能明显缩短时间，同时还能保留大部分趋势
- `10000` 张训练样本：适合更快的对比实验，但结果波动会更大

可参考的改法：

```python
targets = np.array(trainset.targets)
keep = []
samples_per_class = 2000
for cls in range(10):
    cls_idx = np.where(targets == cls)[0]
    keep.extend(cls_idx[:samples_per_class])
trainset = torch.utils.data.Subset(trainset, keep)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
```

说明：

- 这比“直接截前 N 张图片”更稳，因为它保留了类别平衡。
- 如果只是想看训练是否正常，`samples_per_class = 500` 也可以，但那更像 smoke test，不适合拿来下正式结论。

### 方案 B：把训练轮次从 300 降到 80 或 100

修改位置：

- [CIARD.py](F:\Softmax-Distillation\CIARD\CIARD.py) 第 33 行

推荐值：

- `epochs = 100`：更稳，通常更适合“保留结论趋势”
- `epochs = 80`：更快，适合先做一轮验证

但是这里有一个非常重要的同步修改：

- 第 264-269 行的学生学习率日程使用了固定阈值 `150` 和 `300`
- 第 274-281 行的 teacher 学习率日程使用了固定阈值 `50` 和 `300`

如果你只改 `epochs`，不改这些阈值，学习率计划会失真，训练结论会更容易被影响。

建议按比例缩放：

- 原学生衰减起点：`150 / 300 = 0.5`
- 原 teacher 启动点：`50 / 300 ≈ 0.167`

例如：

- 如果 `epochs = 100`
  - 学生衰减起点改成 `50`
  - teacher 启动点改成 `17`
- 如果 `epochs = 80`
  - 学生衰减起点改成 `40`
  - teacher 启动点改成 `13`

可参考的思路：

```python
epochs = 100
student_decay_start = 50
teacher_start_epoch = 17
```

然后把下面这些判断一起改掉：

```python
if epoch < 150:
```

改成：

```python
if epoch < student_decay_start:
```

以及：

```python
np.cos(np.pi * (epoch - 150) / (300 - 150))
```

改成：

```python
np.cos(np.pi * (epoch - student_decay_start) / (epochs - student_decay_start))
```

teacher 那段同理，把 `50` 和 `300` 替换成 `teacher_start_epoch` 和 `epochs`。

### 方案 C：减少中间评估频率，保留最终评估

修改位置：

- [CIARD.py](F:\Softmax-Distillation\CIARD\CIARD.py) 第 303 行附近

当前逻辑：

```python
if epoch == 1 or epoch%10==  0 or epoch >= 250:
```

这意味着：

- 第 1 个 epoch 就评估一次
- 之后每 10 个 epoch 评估一次
- 最后 50 个 epoch 每个 epoch 都评估一次

而每次评估都包含 PGD-20，所以非常耗时。

更省时、又比较稳妥的做法：

```python
if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
```

这样可以：

- 保留初始观测点
- 中间只做稀疏评估
- 最后保留一次完整结果

如果你把 `epochs` 改成了 `80` 或 `100`，这一条通常能再省下不少时间。

## 四、可选的中风险缩时方案

下面这些也能提速，但比上面三项更可能影响鲁棒精度，建议只在你已经确认主流程可跑之后再用。

### 方案 D：减少训练阶段的 PGD 步数

修改位置：

- [CIARD.py](F:\Softmax-Distillation\CIARD\CIARD.py) 第 208-212 行

当前：

```python
perturb_steps=10
```

可选：

- 改成 `7`：通常是一个相对温和的提速
- 改成 `5`：提速更明显，但对鲁棒训练强度影响更大

建议：

- 如果你的目标是“尽量不影响结论”，优先先改数据量、epoch、评估频率
- `perturb_steps` 放到最后再考虑

### 方案 E：只在最终阶段做重评估

训练脚本里当前测试使用：

- [CIARD.py](F:\Softmax-Distillation\CIARD\CIARD.py) 第 325 行：`attack_iters=20`

更稳妥的做法是：

- 中间评估用 `attack_iters=5` 或 `10`
- 最终评估保留 `20`

这样中间只看趋势，最后再看正式指标。

## 五、attack_eval.py 的建议

如果你训练后还要跑 [attack_eval.py](F:\Softmax-Distillation\CIARD\attack_eval.py)，建议区分“日常检查”和“最终评估”。

### 日常检查

保留：

- clean acc
- 一种 PGD 评估

先跳过：

- AutoAttack（第 101-102 行）
- Square Attack（第 218 行）
- CW（第 241 行）

原因：

- 它们都很慢
- 对日常判断“模型是否正常收敛”不是必须

### 最终评估

当你选定最终 checkpoint 后，再恢复：

- AutoAttack
- PGD-20
- Square Attack
- CW

这样更符合“开发时快、收尾时严”的节奏。

## 六、我最推荐的一组具体配置

如果你的目标是“先把实验时间压下来，但还想保留相对可信的趋势”，我建议先试这组：

- 训练集：每类 `2000` 张，总计 `20000` 张
- 测试集：保持完整 `10000` 张不变
- `epochs = 100`
- 学生学习率衰减起点：`50`
- teacher 启动 epoch：`17`
- 中间评估频率：每 `20` 个 epoch 一次
- 中间评估 PGD 步数：`10`
- 最终评估 PGD 步数：`20`
- AutoAttack：只对最终模型运行

这组改法通常比原始设置快很多，而且不会像“大幅改模型结构或改攻击预算”那样直接破坏可比性。

## 七、如果你只是想快速确认代码能跑

这不是“保留结论”的方案，只适合 debug：

- 训练集：每类 `500` 张
- 测试集：每类 `100` 张或完整测试集
- `epochs = 5` 到 `10`
- `perturb_steps = 3`
- 不跑 AutoAttack

它的价值是：

- 快速确认是否会报错
- 看 loss 和 accuracy 是否大致正常

它的局限是：

- 不能代表正式结论

## 八、建议的修改顺序

为了尽量稳妥，建议按这个顺序逐步改：

1. 只缩小训练集，保留完整测试集。
2. 把 `epochs` 改成 `100`，并同步改学习率日程。
3. 降低中间评估频率。
4. 只在最终模型上跑完整 `attack_eval.py`。
5. 如果还是太慢，再考虑把 `perturb_steps` 从 `10` 降到 `7` 或 `5`。

## 九、一个经验判断

如果你的目标是“论文级复现”，最终还是建议至少做一次：

- 完整测试集
- 完整较强攻击评估
- 固定随机种子

而缩小数据集、减少 epoch 更适合：

- 开发期验证
- 消融前的快速试跑
- 确认代码逻辑无误

