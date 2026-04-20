"""
mobilenet_v2
CIARD
push according to label
consist are decided by top1 prediction
Lr stage decay
"""
# 归一化处理
import importlib.util
import math
import inspect
import os
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from loguru import logger
from torchvision import transforms

from cifar10_models import mobilenet_v2, wideresnet
from cifar10_nat_teacher_models import cifar10_resnet56
from mtard_loss import attack_pgd, robust_inner_loss_push


PREFIX = "Cifar10_MobileNetV2"
DRAW_FILE = PREFIX
MODEL_DIR = os.path.join(".", "model", PREFIX)
logger.add("outputs.txt", encoding="utf-8", mode="w")
EPOCHS = 300
BATCH_SIZE = int(os.getenv("CIARD_BATCH_SIZE", "128"))
EVAL_BATCH_SIZE = int(os.getenv("CIARD_EVAL_BATCH_SIZE", str(BATCH_SIZE * 2)))
EPSILON = 8 / 255.0
EVAL_INTERVAL = 20
BASELINE_EPOCHS = 300
DATALOADER_WORKERS = int(
    os.getenv("CIARD_NUM_WORKERS", str(min(8, max(1, (os.cpu_count() or 1) // 2))))
)
PIN_MEMORY = os.getenv("CIARD_PIN_MEMORY", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
    "off",
}
PERSISTENT_WORKERS = os.getenv("CIARD_PERSISTENT_WORKERS", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
    "off",
}
PREFETCH_FACTOR = max(2, int(os.getenv("CIARD_PREFETCH_FACTOR", "4")))
PREFETCH_TO_DEVICE = os.getenv("CIARD_PREFETCH_TO_DEVICE", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
    "off",
}
USE_CHANNELS_LAST = os.getenv("CIARD_USE_CHANNELS_LAST", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
    "off",
}
ENABLE_AMP = os.getenv("CIARD_ENABLE_AMP", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
    "off",
}
AMP_DTYPE_NAME = os.getenv("CIARD_AMP_DTYPE", "auto").strip().lower()
ENABLE_COMPILE = os.getenv("CIARD_ENABLE_COMPILE", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
    "off",
}
COMPILE_BACKEND = os.getenv("CIARD_COMPILE_BACKEND", "inductor")
COMPILE_MODE = os.getenv("CIARD_COMPILE_MODE", "default")
COMPILE_DYNAMIC = os.getenv("CIARD_COMPILE_DYNAMIC", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
    "off",
}
COMPILE_DISABLE_CUDAGRAPHS = os.getenv(
    "CIARD_COMPILE_DISABLE_CUDAGRAPHS",
    "1",
).strip().lower() not in {
    "",
    "0",
    "false",
    "no",
    "off",
}
TRAIN_DROP_LAST = os.getenv(
    "CIARD_TRAIN_DROP_LAST",
    "1" if ENABLE_COMPILE else "0",
).strip().lower() not in {
    "",
    "0",
    "false",
    "no",
    "off",
}
ENABLE_KL_LOSS4 = False  # 引入klloss4
ENABLE_IGDM_ALIGNMENT = False  # 引入梯度对齐
ENABLE_IGDM_FEATURE_ALIGNMENT = False  # 引入中间层间接梯度对齐
IGDM_ALIGN_ALPHA = float(
    os.getenv("CIARD_IGDM_TARGET_RATIO", "0.3")
)  # Increase the default IGDM contribution budget so logit-difference alignment has a stronger effect.
IGDM_START_MARKER = int(os.getenv("CIARD_IGDM_START_MARKER", "1"))
IGDM_RAMP_MARKER = int(os.getenv("CIARD_IGDM_RAMP_MARKER", "60"))
IGDM_CONFIDENCE_THRESHOLD = float(os.getenv("CIARD_IGDM_CONFIDENCE", "0.70"))
IGDM_MARGIN_THRESHOLD = float(os.getenv("CIARD_IGDM_MARGIN", "0.20"))
IGDM_MAX_SCALE = float(os.getenv("CIARD_IGDM_MAX_SCALE", "5.0"))
IGDM_FEATURE_ALIGN_BETA = float(os.getenv("CIARD_IGDM_FEATURE_BETA", "0.35"))
PUSH_ADV_RATIO = float(os.getenv("CIARD_PUSH_ADV_RATIO", "0.6"))
PUSH_MIN_RATIO = float(os.getenv("CIARD_PUSH_MIN_RATIO", "0.3"))
PUSH_IGDM_RELIEF = float(os.getenv("CIARD_PUSH_IGDM_RELIEF", "0.5"))
PUSH_MAX_SCALE = float(os.getenv("CIARD_PUSH_MAX_SCALE", "10.0"))

RESUME_STUDENT_PATH = None
TEACHER1_PATH = "models/model_cifar_wrn.pt"
TEACHER2_PATH = "models/nat_teacher_checkpoint/cifar10_resnnet56.pth"
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError(
            "CIARD requires an NVIDIA CUDA GPU. For RTX 5070 Ti, follow setup_rtx5070ti.md."
        )
    return device


def configure_runtime(device):
    # Keep the seed fixed while allowing cuDNN to pick faster kernels.
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if (
            ENABLE_COMPILE
            and COMPILE_BACKEND == "inductor"
            and COMPILE_DISABLE_CUDAGRAPHS
            and hasattr(torch, "_inductor")
            and hasattr(torch._inductor, "config")
            and hasattr(torch._inductor.config, "triton")
            and hasattr(torch._inductor.config.triton, "cudagraphs")
        ):
            torch._inductor.config.triton.cudagraphs = False


def resolve_compile_options():
    if COMPILE_BACKEND == "inductor" and COMPILE_DISABLE_CUDAGRAPHS:
        return {"triton.cudagraphs": False}
    return None


def should_retry_compile_without_options(exc):
    message = str(exc)
    return (
        "Either mode or options can be specified" in message
        or "unexpected keyword argument 'options'" in message
        or 'unexpected keyword argument "options"' in message
    )


def compile_callable(target, dynamic):
    compile_kwargs = {
        "backend": COMPILE_BACKEND,
        "mode": COMPILE_MODE,
        "dynamic": dynamic,
    }
    compile_options = resolve_compile_options()
    if compile_options is not None:
        compile_kwargs["options"] = compile_options
    try:
        return torch.compile(target, **compile_kwargs)
    except Exception as exc:
        if "options" not in compile_kwargs or not should_retry_compile_without_options(exc):
            raise
        logger.warning(
            "torch.compile cannot use mode and options together in this PyTorch build. "
            "Retrying without explicit compile options: {}",
            exc,
        )
        compile_kwargs.pop("options", None)
        return torch.compile(target, **compile_kwargs)


def resolve_amp_dtype(device):
    if device.type != "cuda" or not ENABLE_AMP:
        return None
    if AMP_DTYPE_NAME in {"fp16", "float16", "half"}:
        return torch.float16
    if AMP_DTYPE_NAME in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if AMP_DTYPE_NAME != "auto":
        raise ValueError(f"Unsupported CIARD_AMP_DTYPE value: {AMP_DTYPE_NAME}")
    bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    return torch.bfloat16 if bf16_supported else torch.float16


def autocast_context(device, amp_dtype):
    if device.type != "cuda" or amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def build_grad_scaler(device, amp_dtype):
    scaler_enabled = device.type == "cuda" and amp_dtype == torch.float16
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type, enabled=scaler_enabled)
    return torch.cuda.amp.GradScaler(enabled=scaler_enabled)


def unwrap_model(model):
    return getattr(model, "_orig_mod", model)


def is_compiled_model(model):
    return unwrap_model(model) is not model


def materialize_tensor(tensor):
    if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
        return tensor.clone()
    return tensor


def materialize_model_output(model, output):
    if is_compiled_model(model):
        return materialize_tensor(output)
    return output


def mark_compile_step_begin(compile_enabled):
    if (
        compile_enabled
        and hasattr(torch, "compiler")
        and hasattr(torch.compiler, "cudagraph_mark_step_begin")
    ):
        torch.compiler.cudagraph_mark_step_begin()


def maybe_channels_last(model):
    if USE_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    return model


def get_compile_status(device):
    if not ENABLE_COMPILE:
        return False, "disabled by CIARD_ENABLE_COMPILE"
    if device.type != "cuda":
        return False, "requires CUDA"
    if not hasattr(torch, "compile"):
        return False, "torch.compile is unavailable in this PyTorch build"
    if COMPILE_BACKEND == "inductor":
        triton_spec = importlib.util.find_spec("triton")
        if triton_spec is None:
            return False, "Triton is not installed"
        try:
            import triton  # noqa: F401
        except Exception as exc:
            return False, f"Triton import failed: {exc}"
    try:
        def compile_probe(x):
            return x + 1

        compiled_probe = compile_callable(compile_probe, dynamic=False)
        probe_input = torch.zeros(8, device=device)
        _ = compiled_probe(probe_input)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    except Exception as exc:
        return False, f"torch.compile probe failed: {exc}"
    return True, "available"


def maybe_compile_model(model, name, device, compile_enabled, compile_reason):
    if not compile_enabled:
        logger.warning(
            "Skipping torch.compile for {}: {}. Training will continue in eager mode.",
            name,
            compile_reason,
        )
        return model
    try:
        compiled_model = compile_callable(model, dynamic=COMPILE_DYNAMIC)
        logger.info(
            "torch.compile enabled for {} with backend={} mode={} dynamic={} "
            "disable_cudagraphs={}",
            name,
            COMPILE_BACKEND,
            COMPILE_MODE,
            COMPILE_DYNAMIC,
            COMPILE_DISABLE_CUDAGRAPHS,
        )
        return compiled_model
    except Exception as exc:
        logger.warning("torch.compile failed for {}. Falling back to eager mode: {}", name, exc)
        return model


def effective_train_drop_last(compile_enabled):
    return TRAIN_DROP_LAST or compile_enabled


def load_checkpoint(path):
    load_kwargs = {"map_location": torch.device("cpu")}
    try:
        return torch.load(path, weights_only=True, **load_kwargs)
    except pickle.UnpicklingError as exc:
        logger.warning(
            "weights_only=True failed for checkpoint {}. Falling back to "
            "weights_only=False because this file is assumed to be trusted. "
            "Original error: {}",
            path,
            exc,
        )
        return torch.load(path, weights_only=False, **load_kwargs)
    except TypeError:
        return torch.load(path, **load_kwargs)


def load_state_dict_from_checkpoint(path, key="model"):
    checkpoint = load_checkpoint(path)
    if isinstance(checkpoint, dict) and key in checkpoint and hasattr(checkpoint[key], "items"):
        checkpoint = checkpoint[key]
    if not hasattr(checkpoint, "items"):
        raise TypeError(f"Unsupported checkpoint format in {path}")
    return {k.replace("module.", ""): v for k, v in checkpoint.items()}


def scale_epoch_marker(marker, total_epochs, reference_epochs=BASELINE_EPOCHS):
    return max(1, min(total_epochs, int(round(marker * total_epochs / reference_epochs))))


class InputNormalize(torch.nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer(
            "mean",
            torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1),
        )

    def forward(self, inputs):
        normalized_inputs = (inputs - self.mean) / self.std
        return self.model(normalized_inputs)


def move_batch_to_device(inputs, labels, device, non_blocking):
    inputs = inputs.to(device, non_blocking=non_blocking)
    if USE_CHANNELS_LAST and inputs.ndim == 4:
        inputs = inputs.to(memory_format=torch.channels_last)
    labels = labels.to(device, non_blocking=non_blocking)
    return inputs, labels


class CUDAPrefetcher:
    def __init__(self, loader, device, non_blocking):
        self.loader = loader
        self.device = device
        self.non_blocking = non_blocking
        self.stream = torch.cuda.Stream(device=device)

    def __iter__(self):
        loader_iter = iter(self.loader)
        next_inputs = None
        next_labels = None

        def preload():
            try:
                batch_inputs, batch_labels = next(loader_iter)
            except StopIteration:
                return None, None
            with torch.cuda.stream(self.stream):
                batch_inputs, batch_labels = move_batch_to_device(
                    batch_inputs,
                    batch_labels,
                    self.device,
                    self.non_blocking,
                )
            return batch_inputs, batch_labels

        next_inputs, next_labels = preload()
        while next_inputs is not None:
            current_stream = torch.cuda.current_stream(device=self.device)
            current_stream.wait_stream(self.stream)
            next_inputs.record_stream(current_stream)
            next_labels.record_stream(current_stream)
            batch_inputs, batch_labels = next_inputs, next_labels
            next_inputs, next_labels = preload()
            yield batch_inputs, batch_labels


def iterate_device_batches(loader, device, non_blocking):
    if device.type == "cuda" and PREFETCH_TO_DEVICE:
        return CUDAPrefetcher(loader, device, non_blocking)

    def generator():
        for batch_inputs, batch_labels in loader:
            yield move_batch_to_device(batch_inputs, batch_labels, device, non_blocking)

    return generator()


def build_dataloaders(device, compile_enabled):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    transform_test = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform_train,
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_test,
    )

    loader_kwargs = {
        "num_workers": DATALOADER_WORKERS,
        "pin_memory": device.type == "cuda" and PIN_MEMORY,
    }
    loader_signature = inspect.signature(torch.utils.data.DataLoader).parameters
    if loader_kwargs["pin_memory"] and "pin_memory_device" in loader_signature:
        loader_kwargs["pin_memory_device"] = str(device)
    if DATALOADER_WORKERS > 0:
        loader_kwargs["persistent_workers"] = PERSISTENT_WORKERS
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    train_drop_last = effective_train_drop_last(compile_enabled)
    if compile_enabled and not TRAIN_DROP_LAST:
        logger.warning(
            "Forcing train DataLoader drop_last=True while torch.compile is enabled "
            "to keep compiled training batch shapes stable."
        )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=train_drop_last,
        **loader_kwargs,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        **loader_kwargs,
    )
    return trainset, testset, trainloader, testloader, train_drop_last


def build_models(device):
    student = mobilenet_v2()
    if RESUME_STUDENT_PATH is not None:
        student.load_state_dict(load_state_dict_from_checkpoint(RESUME_STUDENT_PATH))
    student = maybe_channels_last(student.to(device))
    student.train()

    optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
    begin_epoch = 1 if RESUME_STUDENT_PATH is None else scale_epoch_marker(200, EPOCHS)

    teacher = wideresnet()
    teacher.load_state_dict(load_state_dict_from_checkpoint(TEACHER1_PATH, key="model"))
    teacher = maybe_channels_last(teacher.to(device))
    teacher.train()

    teacher_lr = 0.0001
    adv_teacher_optimizer = optim.SGD(
        teacher.parameters(),
        lr=teacher_lr,
        momentum=0.1,
        weight_decay=2e-4,
    )
    adv_teacher_loss_ce = torch.nn.CrossEntropyLoss().to(device)

    teacher_nat_model = cifar10_resnet56()
    teacher_nat_model.load_state_dict(
        load_state_dict_from_checkpoint(TEACHER2_PATH, key="model")
    )
    teacher_nat_model = maybe_channels_last(teacher_nat_model.to(device))
    teacher_nat = maybe_channels_last(
        InputNormalize(teacher_nat_model, CIFAR10_MEAN, CIFAR10_STD).to(device)
    )
    teacher_nat.eval()

    ce_loss = torch.nn.CrossEntropyLoss().to(device)
    return (
        student,
        teacher,
        teacher_nat,
        optimizer,
        adv_teacher_optimizer,
        adv_teacher_loss_ce,
        ce_loss,
        begin_epoch,
    )


def prepare_output_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, DRAW_FILE), "w", encoding="utf-8") as handle:
        handle.write(
            "epoch student_robust_acc student_natural_acc "
            "adv_teacher_robust_acc adv_teacher_natural_acc "
            "nat_teacher_robust_acc nat_teacher_natural_acc\n"
        )


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class FeatureMonitor:
    def __init__(self, module):
        self.output = None
        self.handle = module.register_forward_hook(self._save_output)

    def _save_output(self, _module, _inputs, output):
        self.output = output

    def close(self):
        self.handle.remove()


def resolve_igdm_feature_module(model, role):
    model = unwrap_model(model)
    if role == "student":
        candidate_names = ("layers", "features", "layer4")
    else:
        candidate_names = ("block3", "layer4", "features")
    for name in candidate_names:
        module = getattr(model, name, None)
        if module is not None:
            return module
    raise ValueError(
        f"Unable to resolve an IGDM feature module for {role}: {type(model).__name__}"
    )


def build_igdm_feature_monitors(student, teacher):
    return {
        "student": FeatureMonitor(resolve_igdm_feature_module(student, "student")),
        "teacher": FeatureMonitor(resolve_igdm_feature_module(teacher, "robust_teacher")),
    }


def kl_loss(a, b):
    return -a * b + torch.log(b + 1e-5) * b


def entropy_value(a):
    return torch.log(a + 1e-5) * a


def scale_to_magnitude(a, b, c):
    if math.isclose(a, 0, rel_tol=1e-9):
        a += 1e-7
    if math.isclose(b, 0, rel_tol=1e-9):
        b += 1e-7
    if math.isclose(c, 0, rel_tol=1e-9):
        c += 1e-7
    magnitude_a = math.floor(math.log10(abs(a)))
    magnitude_b = math.floor(math.log10(abs(b)))
    target_magnitude = min(magnitude_a, magnitude_b)
    magnitude_c = math.floor(math.log10(abs(c)))
    return 10 ** (target_magnitude - magnitude_c)


def push_loss(teacher_logits, student_logits, labels, temperature=5):
    teacher_predictions = torch.argmax(teacher_logits, dim=1)
    diff_indices = (teacher_predictions != labels).nonzero(as_tuple=True)[0]
    diff_teacher_logits = teacher_logits[diff_indices]
    diff_student_logits = student_logits[diff_indices]
    return kl_loss(
        F.log_softmax(diff_student_logits / temperature, dim=1),
        F.softmax(diff_teacher_logits.detach(), dim=1),
    )


def forward_eval_logits(model, inputs):
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        logits = model(inputs)
    logits = materialize_model_output(model, logits)
    if was_training:
        model.train()
    return logits


def forward_with_mode(model, inputs, training):
    was_training = model.training
    if was_training != training:
        model.train(training)
    outputs = model(inputs)
    outputs = materialize_model_output(model, outputs)
    if was_training != training:
        model.train(was_training)
    return outputs


def forward_with_feature_capture(model, inputs, monitor=None, training=False, no_grad=False):
    was_training = model.training
    if was_training != training:
        model.train(training)
    if no_grad:
        with torch.inference_mode():
            outputs = model(inputs)
    else:
        outputs = model(inputs)
    outputs = materialize_model_output(model, outputs)
    feature = materialize_tensor(monitor.output) if monitor is not None else None
    if was_training != training:
        model.train(was_training)
    return outputs, feature


def linear_rampup(epoch, start_epoch, ramp_epochs):
    if epoch < start_epoch:
        return 0.0
    return min(1.0, (epoch - start_epoch + 1) / max(1, ramp_epochs))


def build_igdm_confidence_mask(
    teacher_logits,
    labels,
    confidence_threshold=IGDM_CONFIDENCE_THRESHOLD,
    margin_threshold=IGDM_MARGIN_THRESHOLD,
):
    teacher_probs = F.softmax(teacher_logits.detach(), dim=1)
    top2 = torch.topk(teacher_probs, k=min(2, teacher_probs.size(1)), dim=1).values
    confidence = top2[:, 0]
    margin = confidence if top2.size(1) == 1 else confidence - top2[:, 1]
    predictions = torch.argmax(teacher_probs, dim=1)
    mask = (
        (predictions == labels)
        & (confidence >= confidence_threshold)
        & (margin >= margin_threshold)
    )
    return mask, confidence, margin


def ratio_controlled_weight(reference_loss, aux_loss, target_ratio, max_scale):
    if target_ratio <= 0:
        return 0.0
    reference_value = float(reference_loss.detach().item())
    aux_value = float(aux_loss.detach().item())
    if not math.isfinite(reference_value) or not math.isfinite(aux_value):
        return 0.0
    if reference_value <= 0 or aux_value <= 0:
        return 0.0
    return min(max_scale, target_ratio * reference_value / (aux_value + 1e-8))


def attention_transfer_map(feature_map, output_size):
    if feature_map.shape[-2:] != output_size:
        feature_map = F.adaptive_avg_pool2d(feature_map, output_size)
    attention = feature_map.pow(2).mean(dim=1)
    attention = attention.flatten(1)
    return F.normalize(attention, dim=1, eps=1e-6)


def indirect_gradient_feature_alignment_loss(
    student_plus_feature,
    student_minus_feature,
    teacher_plus_feature,
    teacher_minus_feature,
):
    feature_tensors = (
        student_plus_feature,
        student_minus_feature,
        teacher_plus_feature,
        teacher_minus_feature,
    )
    if any(feature is None for feature in feature_tensors):
        for feature in feature_tensors:
            if feature is not None:
                return torch.tensor(0.0, device=feature.device)
        return torch.tensor(0.0, device=torch.device("cpu"))

    target_size = student_plus_feature.shape[-2:]
    student_plus_attention = attention_transfer_map(student_plus_feature, target_size)
    student_minus_attention = attention_transfer_map(student_minus_feature, target_size)
    teacher_plus_attention = attention_transfer_map(
        teacher_plus_feature.detach(),
        target_size,
    )
    teacher_minus_attention = attention_transfer_map(
        teacher_minus_feature.detach(),
        target_size,
    )
    student_diff = student_plus_attention - student_minus_attention
    teacher_diff = teacher_plus_attention - teacher_minus_attention
    return F.mse_loss(student_diff, teacher_diff)


def igdm_alignment_losses(
    student,
    teacher,
    x_natural,
    x_adv,
    student_adv_logits,
    epsilon,
    feature_monitors=None,
    sample_mask=None,
    student_plus_feature=None,
):
    if sample_mask is None:
        sample_mask = torch.ones(
            x_natural.size(0),
            dtype=torch.bool,
            device=x_natural.device,
        )
    if sample_mask.sum().item() == 0:
        zero = torch.tensor(0.0, device=x_natural.device)
        return zero, zero

    delta = torch.clamp(x_adv - x_natural, min=-epsilon, max=epsilon)
    x_minus = torch.clamp(x_natural - delta, 0.0, 1.0)

    student_minus_logits = forward_with_mode(student, x_minus, training=True)
    teacher_plus_logits = forward_eval_logits(teacher, x_adv)
    teacher_minus_logits = forward_eval_logits(teacher, x_minus)

    student_diff = student_adv_logits - student_minus_logits
    teacher_diff = teacher_plus_logits - teacher_minus_logits
    logit_loss = F.mse_loss(student_diff[sample_mask], teacher_diff[sample_mask])
    feature_loss = torch.tensor(0.0, device=x_natural.device)
    if ENABLE_IGDM_FEATURE_ALIGNMENT and feature_monitors:
        student_monitor = feature_monitors.get("student")
        teacher_monitor = feature_monitors.get("teacher")
        if student_plus_feature is None:
            _, student_plus_feature = forward_with_feature_capture(
                student,
                x_adv,
                monitor=student_monitor,
                training=True,
            )
        student_plus_feature = student_plus_feature[sample_mask]
        _, student_minus_feature = forward_with_feature_capture(
            student,
            x_minus,
            monitor=student_monitor,
            training=True,
        )
        student_minus_feature = student_minus_feature[sample_mask]
        _, teacher_plus_feature = forward_with_feature_capture(
            teacher,
            x_adv,
            monitor=teacher_monitor,
            training=False,
            no_grad=True,
        )
        teacher_plus_feature = teacher_plus_feature[sample_mask]
        _, teacher_minus_feature = forward_with_feature_capture(
            teacher,
            x_minus,
            monitor=teacher_monitor,
            training=False,
            no_grad=True,
        )
        teacher_minus_feature = teacher_minus_feature[sample_mask]
        feature_loss = indirect_gradient_feature_alignment_loss(
            student_plus_feature,
            student_minus_feature,
            teacher_plus_feature,
            teacher_minus_feature,
        )
    return logit_loss, feature_loss


def clean_push_loss(robust_teacher_logits, natural_teacher_logits, student_logits, labels, temperature=5):
    robust_teacher_predictions = torch.argmax(robust_teacher_logits, dim=1)
    natural_teacher_predictions = torch.argmax(natural_teacher_logits, dim=1)
    diff_indices = (
        (robust_teacher_predictions != labels) & (natural_teacher_predictions == labels)
    ).nonzero(as_tuple=True)[0]
    diff_robust_teacher_logits = robust_teacher_logits[diff_indices]
    diff_student_logits = student_logits[diff_indices]
    return kl_loss(
        F.log_softmax(diff_student_logits / temperature, dim=1),
        F.softmax(diff_robust_teacher_logits.detach(), dim=1),
    )


def log_runtime_settings(device, amp_dtype, compile_enabled, compile_reason, train_drop_last):
    amp_dtype_name = str(amp_dtype).replace("torch.", "") if amp_dtype is not None else "disabled"
    logger.info(
        "runtime config: batch_size={}, eval_batch_size={}, num_workers={}, "
        "pin_memory={}, persistent_workers={}, prefetch_factor={}, train_drop_last={}, "
        "prefetch_to_device={}, channels_last={}, amp_dtype={}, "
        "compile_enabled={}, compile_reason={}, compile_backend={}, compile_mode={}, compile_dynamic={}, "
        "compile_disable_cudagraphs={}, eval_uses_eager={}, pgd_uses_eager={}, "
        "tf32={}, cudnn_benchmark={}, enable_kl_loss4={}, "
        "enable_igdm_alignment={}, enable_igdm_feature_alignment={}, "
        "igdm_target_ratio={}, igdm_start_marker={}, igdm_ramp_marker={}, "
        "igdm_confidence_threshold={}, igdm_margin_threshold={}, "
        "igdm_max_scale={}, igdm_feature_align_beta={}, push_adv_ratio={}, "
        "push_min_ratio={}, push_igdm_relief={}, push_max_scale={}",
        BATCH_SIZE,
        EVAL_BATCH_SIZE,
        DATALOADER_WORKERS,
        device.type == "cuda" and PIN_MEMORY,
        PERSISTENT_WORKERS if DATALOADER_WORKERS > 0 else False,
        PREFETCH_FACTOR if DATALOADER_WORKERS > 0 else 0,
        train_drop_last,
        PREFETCH_TO_DEVICE and device.type == "cuda",
        USE_CHANNELS_LAST,
        amp_dtype_name,
        compile_enabled,
        compile_reason,
        COMPILE_BACKEND,
        COMPILE_MODE,
        COMPILE_DYNAMIC,
        COMPILE_DISABLE_CUDAGRAPHS,
        True,
        True,
        torch.backends.cuda.matmul.allow_tf32 if device.type == "cuda" else False,
        torch.backends.cudnn.benchmark,
        ENABLE_KL_LOSS4,
        ENABLE_IGDM_ALIGNMENT,
        ENABLE_IGDM_FEATURE_ALIGNMENT,
        IGDM_ALIGN_ALPHA,
        IGDM_START_MARKER,
        IGDM_RAMP_MARKER,
        IGDM_CONFIDENCE_THRESHOLD,
        IGDM_MARGIN_THRESHOLD,
        IGDM_MAX_SCALE,
        IGDM_FEATURE_ALIGN_BETA,
        PUSH_ADV_RATIO,
        PUSH_MIN_RATIO,
        PUSH_IGDM_RELIEF,
        PUSH_MAX_SCALE,
    )


def main():
    device = get_device()
    configure_runtime(device)
    prepare_output_dir()
    amp_dtype = resolve_amp_dtype(device)
    scaler = build_grad_scaler(device, amp_dtype)
    compile_enabled, compile_reason = get_compile_status(device)

    trainset, testset, trainloader, testloader, train_drop_last = build_dataloaders(
        device,
        compile_enabled,
    )
    (
        student,
        teacher,
        teacher_nat,
        optimizer,
        adv_teacher_optimizer,
        adv_teacher_loss_ce,
        ce_loss,
        begin_epoch,
    ) = build_models(device)
    student_decay_start = scale_epoch_marker(150, EPOCHS)
    teacher_start_epoch = scale_epoch_marker(50, EPOCHS)
    teacher_update_start = teacher_start_epoch
    feature_monitors = (
        build_igdm_feature_monitors(student, teacher)
        if ENABLE_IGDM_ALIGNMENT and ENABLE_IGDM_FEATURE_ALIGNMENT
        else {}
    )
    student = maybe_compile_model(
        student,
        "student",
        device,
        compile_enabled,
        compile_reason,
    )
    teacher = maybe_compile_model(
        teacher,
        "teacher",
        device,
        compile_enabled,
        compile_reason,
    )
    teacher_nat = maybe_compile_model(
        teacher_nat,
        "teacher_nat",
        device,
        compile_enabled,
        compile_reason,
    )
    student_eval = unwrap_model(student)
    teacher_eval = unwrap_model(teacher)
    teacher_nat_eval = unwrap_model(teacher_nat)
    igdm_start_epoch = scale_epoch_marker(IGDM_START_MARKER, EPOCHS)
    igdm_ramp_epochs = scale_epoch_marker(IGDM_RAMP_MARKER, EPOCHS)
    adaptive_weight_decay_epochs = sorted(
        set(scale_epoch_marker(marker, EPOCHS) for marker in (215, 260, 285))
    )
    latest_checkpoint_start = scale_epoch_marker(250, EPOCHS)

    weight = {"adv_loss": 0.5, "nat_loss": 0.5}
    init_loss_nat = None
    init_loss_adv = None
    best_accuracy = 0
    weight_learn_rate = 0.025
    temp_learn_rate = 0.001
    temp_adv = 1
    temp_nat = 1
    temp_max = 10
    temp_min = 1
    non_blocking = device.type == "cuda"

    logger.info(
        """
CIARD
push label T=5
Lr stage decay
push_loss(nat_adv_logits,student_adv_logits,train_batch_labels)
teacher lr weight decay from 0.0001 to 0 with smooth decay
epoch = {} coslr
train dataset = full CIFAR-10 training set
student decay start = {}
teacher start epoch = {}
igdm start epoch = {}
igdm ramp epochs = {}
enable kl_loss4 = {}
enable igdm feature alignment = {}
""".format(
            EPOCHS,
            student_decay_start,
            teacher_start_epoch,
            igdm_start_epoch,
            igdm_ramp_epochs,
            ENABLE_KL_LOSS4,
            ENABLE_IGDM_FEATURE_ALIGNMENT,
        )
    )
    logger.info(
        "using {} training samples and {} test samples",
        len(trainset),
        len(testset),
    )
    log_runtime_settings(
        device,
        amp_dtype,
        compile_enabled,
        compile_reason,
        train_drop_last,
    )

    for epoch in range(begin_epoch, EPOCHS + 1):
        logger.info("the {}th epoch ", epoch)
        for step, (train_batch_data, train_batch_labels) in enumerate(
            iterate_device_batches(trainloader, device, non_blocking)
        ):
            mark_compile_step_begin(compile_enabled)
            student.train()
            teacher.train()

            with autocast_context(device, amp_dtype):
                student_nat_logits = materialize_model_output(
                    student,
                    student(train_batch_data),
                )
                adv_teacher_nat_logits = None
                with torch.inference_mode():
                    teacher_nat_logits = materialize_model_output(
                        teacher_nat,
                        teacher_nat(train_batch_data),
                    )
                if ENABLE_KL_LOSS4:
                    adv_teacher_nat_logits = forward_eval_logits(teacher, train_batch_data)

            student_adv_logits, teacher_adv_logits, nat_adv_logits, x_adv = robust_inner_loss_push(
                student,
                teacher,
                teacher_nat,
                train_batch_data,
                train_batch_labels,
                optimizer,
                adv_teacher_optimizer,
                step_size=2 / 255.0,
                epsilon=EPSILON,
                perturb_steps=10,
                amp_enabled=amp_dtype is not None,
                amp_dtype=amp_dtype,
            )
            student_adv_logits = materialize_model_output(student, student_adv_logits)
            teacher_adv_logits = materialize_model_output(teacher, teacher_adv_logits)
            nat_adv_logits = materialize_model_output(teacher_nat, nat_adv_logits)
            student_adv_feature = (
                materialize_tensor(feature_monitors["student"].output)
                if feature_monitors
                else None
            )
            student_nat_logits_for_loss = student_nat_logits.float()
            teacher_nat_logits_for_loss = teacher_nat_logits.float()
            student_adv_logits_for_loss = student_adv_logits.float()
            teacher_adv_logits_for_loss = teacher_adv_logits.float()
            nat_adv_logits_for_loss = nat_adv_logits.float()
            adv_teacher_nat_logits_for_loss = (
                adv_teacher_nat_logits.float() if adv_teacher_nat_logits is not None else None
            )

            kl_loss1 = kl_loss(
                F.log_softmax(student_adv_logits_for_loss, dim=1),
                F.softmax(teacher_adv_logits_for_loss.detach() / temp_adv, dim=1),
            )
            kl_loss2 = kl_loss(
                F.log_softmax(student_nat_logits_for_loss, dim=1),
                F.softmax(teacher_nat_logits_for_loss.detach() / temp_nat, dim=1),
            )
            kl_loss1 = torch.mean(kl_loss1)
            kl_loss2 = torch.mean(kl_loss2)
            adv_teacher_entropy = torch.mean(
                entropy_value(F.softmax(teacher_adv_logits_for_loss.detach() / temp_adv, dim=1))
            )
            nat_teacher_entropy = torch.mean(
                entropy_value(F.softmax(teacher_nat_logits_for_loss.detach() / temp_nat, dim=1))
            )

            temp_adv = temp_adv - temp_learn_rate * torch.sign(
                adv_teacher_entropy.detach() / nat_teacher_entropy.detach() - 1
            ).item()
            temp_nat = temp_nat - temp_learn_rate * torch.sign(
                nat_teacher_entropy.detach() / adv_teacher_entropy.detach() - 1
            ).item()
            temp_adv = max(min(temp_max, temp_adv), temp_min)
            temp_nat = max(min(temp_max, temp_nat), temp_min)

            if init_loss_nat is None:
                init_loss_nat = kl_loss2.item()
            if init_loss_adv is None:
                init_loss_adv = kl_loss1.item()

            lhat_adv = kl_loss1.item() / init_loss_adv
            lhat_nat = kl_loss2.item() / init_loss_nat
            lhat_avg = (lhat_adv + lhat_nat) / len(weight)
            inv_rate_adv = lhat_adv / lhat_avg
            inv_rate_nat = lhat_nat / lhat_avg

            weight["nat_loss"] = weight["nat_loss"] - weight_learn_rate * (
                weight["nat_loss"] - inv_rate_nat / (inv_rate_adv + inv_rate_nat)
            )
            weight["adv_loss"] = weight["adv_loss"] - weight_learn_rate * (
                weight["adv_loss"] - inv_rate_adv / (inv_rate_adv + inv_rate_nat)
            )
            if weight["adv_loss"] < 0:
                weight["adv_loss"] = 0
            if weight["nat_loss"] < 0:
                weight["nat_loss"] = 0

            coef = 1.0 / (weight["adv_loss"] + weight["nat_loss"])
            weight["adv_loss"] *= coef
            weight["nat_loss"] *= coef

            # Apply the learned clean/robust balancing weights to the two KD terms.
            total_loss = weight["adv_loss"] * kl_loss1 + weight["nat_loss"] * kl_loss2
            loss3_weight = 0.0
            raw_loss3_weight = 0.0
            loss4_weight = 0.0
            igdm_logit_loss = torch.tensor(0.0, device=device)
            igdm_feature_loss = torch.tensor(0.0, device=device)
            igdm_loss = torch.tensor(0.0, device=device)
            igdm_align_weight = 0.0
            igdm_active_ratio = 0.0
            igdm_ramp = (
                linear_rampup(epoch, igdm_start_epoch, igdm_ramp_epochs)
                if ENABLE_IGDM_ALIGNMENT
                else 0.0
            )
            push_ratio_budget = PUSH_ADV_RATIO
            push_weight_cap = 0.0
            robust_reference_loss = (weight["adv_loss"] * kl_loss1).detach()
            kl_loss3 = push_loss(
                nat_adv_logits_for_loss,
                student_adv_logits_for_loss,
                train_batch_labels,
            )
            kl_loss4 = torch.tensor(0.0, device=device)
            if torch.isnan(kl_loss3).any() or kl_loss3.numel() == 0:
                kl_loss3 = torch.tensor(0.0, device=device)
            else:
                kl_loss3 = torch.mean(kl_loss3)
                raw_loss3_weight = scale_to_magnitude(
                    float(kl_loss1.item()),
                    float(kl_loss2.item()),
                    float(kl_loss3.item()),
                )

            if ENABLE_KL_LOSS4 and adv_teacher_nat_logits_for_loss is not None:
                kl_loss4 = clean_push_loss(
                    adv_teacher_nat_logits_for_loss,
                    teacher_nat_logits_for_loss,
                    student_nat_logits_for_loss,
                    train_batch_labels,
                )
                if torch.isnan(kl_loss4).any() or kl_loss4.numel() == 0:
                    kl_loss4 = torch.tensor(0.0, device=device)
                else:
                    kl_loss4 = torch.mean(kl_loss4)
                    loss4_weight = scale_to_magnitude(
                        float(kl_loss1.item()),
                        float(kl_loss2.item()),
                        float(kl_loss4.item()),
                    )
                    total_loss -= loss4_weight * kl_loss4

            if ENABLE_IGDM_ALIGNMENT and igdm_ramp > 0:
                igdm_mask, _, _ = build_igdm_confidence_mask(
                    teacher_adv_logits_for_loss,
                    train_batch_labels,
                )
                igdm_active_ratio = float(igdm_mask.float().mean().item())
                if igdm_mask.sum().item() > 0:
                    with autocast_context(device, amp_dtype):
                        igdm_logit_loss, igdm_feature_loss = igdm_alignment_losses(
                            student,
                            teacher,
                            train_batch_data,
                            x_adv,
                            student_adv_logits,
                            EPSILON,
                            feature_monitors=feature_monitors,
                            sample_mask=igdm_mask,
                            student_plus_feature=student_adv_feature,
                        )
                    igdm_logit_loss = igdm_logit_loss.float()
                    igdm_feature_loss = igdm_feature_loss.float()
                    if not torch.isfinite(igdm_logit_loss).all():
                        igdm_logit_loss = torch.tensor(0.0, device=device)
                    if not torch.isfinite(igdm_feature_loss).all():
                        igdm_feature_loss = torch.tensor(0.0, device=device)
                    igdm_feature_weight = (
                        IGDM_FEATURE_ALIGN_BETA
                        if ENABLE_IGDM_FEATURE_ALIGNMENT
                        else 0.0
                    )
                    igdm_loss = igdm_logit_loss + (
                        igdm_feature_weight * igdm_feature_loss
                    )
                    if torch.isfinite(igdm_loss).all() and igdm_loss.item() > 0:
                        igdm_align_weight = ratio_controlled_weight(
                            robust_reference_loss,
                            igdm_loss,
                            IGDM_ALIGN_ALPHA * igdm_ramp,
                            IGDM_MAX_SCALE,
                        )
                        if igdm_align_weight > 0:
                            total_loss += igdm_align_weight * igdm_loss
                    else:
                        igdm_loss = torch.tensor(0.0, device=device)

            if kl_loss3.item() > 0:
                push_ratio_budget = max(
                    PUSH_MIN_RATIO,
                    PUSH_ADV_RATIO * (1.0 - PUSH_IGDM_RELIEF * igdm_ramp),
                )
                push_weight_cap = ratio_controlled_weight(
                    robust_reference_loss,
                    kl_loss3,
                    push_ratio_budget,
                    PUSH_MAX_SCALE,
                )
                if push_weight_cap > 0:
                    loss3_weight = min(raw_loss3_weight, push_weight_cap)
                    total_loss -= loss3_weight * kl_loss3

            if epoch < student_decay_start:
                lr = 0.1
            else:
                student_decay_span = max(1, EPOCHS - student_decay_start)
                cosine_term = 0.5 + 0.5 * np.cos(
                    np.pi * (epoch - student_decay_start) / student_decay_span
                )
                exponential_decay = np.exp(
                    -0.01 * (epoch - student_decay_start) ** 2 / student_decay_span**2
                )
                lr = 0.1 * cosine_term * exponential_decay

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            if epoch < teacher_start_epoch:
                teacher_lr = 0
            else:
                base_lr = 0.0001
                min_lr = 0
                teacher_decay_span = max(1, EPOCHS - teacher_start_epoch)
                cosine_term = 0.5 + 0.5 * np.cos(
                    np.pi * (epoch - teacher_start_epoch) / teacher_decay_span
                )
                exponential_decay = np.exp(
                    -0.01 * (epoch - teacher_start_epoch) ** 2 / teacher_decay_span**2
                )
                teacher_lr = min_lr + (base_lr - min_lr) * cosine_term * exponential_decay

            for param_group in adv_teacher_optimizer.param_groups:
                param_group["lr"] = teacher_lr

            if epoch in adaptive_weight_decay_epochs:
                weight_learn_rate *= 0.1
                temp_learn_rate *= 0.1

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            if epoch > teacher_update_start:
                adv_teacher_loss = adv_teacher_loss_ce(
                    teacher_adv_logits_for_loss,
                    train_batch_labels,
                )
                scaler.scale(adv_teacher_loss).backward()
                scaler.step(adv_teacher_optimizer)
            scaler.update()

            if step % 100 == 0:
                logger.info(
                    "lr:{} weight_nat: {}, nat_loss: {}, weight_adv: {}, adv_loss: {} "
                    "weight-klloss3 {} raw-klloss3 {} push-ratio {} push-cap {} "
                    "Loss3: {} weight-klloss4 {} Loss4: {} "
                    "weight-igdm {} IGDM: {} IGDM-logit: {} IGDM-feature: {} "
                    "IGDM-ramp: {} IGDM-active: {}",
                    lr,
                    weight["nat_loss"],
                    kl_loss2.item(),
                    weight["adv_loss"],
                    kl_loss1.item(),
                    loss3_weight,
                    raw_loss3_weight,
                    push_ratio_budget,
                    push_weight_cap,
                    kl_loss3.item(),
                    loss4_weight,
                    kl_loss4.item(),
                    igdm_align_weight,
                    igdm_loss.item(),
                    igdm_logit_loss.item(),
                    igdm_feature_loss.item(),
                    igdm_ramp,
                    igdm_active_ratio,
                )

        if epoch == 1 or epoch % EVAL_INTERVAL == 0 or epoch == EPOCHS:
            student_eval.eval()
            teacher_eval.eval()
            teacher_nat_eval.eval()

            robust_total = 0
            student_robust_correct = torch.zeros((), device=device, dtype=torch.long)
            teacher_robust_correct = torch.zeros((), device=device, dtype=torch.long)
            nat_teacher_robust_correct = torch.zeros((), device=device, dtype=torch.long)

            for test_batch_data, test_batch_labels in iterate_device_batches(
                testloader, device, non_blocking
            ):
                test_ifgsm_data = attack_pgd(
                    student_eval,
                    test_batch_data,
                    test_batch_labels,
                    attack_iters=20,
                    step_size=0.003,
                    epsilon=8.0 / 255.0,
                )
                with torch.inference_mode(), autocast_context(device, amp_dtype):
                    logits = student_eval(test_ifgsm_data)
                    teacher_logits = teacher_eval(test_ifgsm_data)
                    nat_teacher_logits = teacher_nat_eval(test_ifgsm_data)
                robust_total += test_batch_labels.size(0)
                student_robust_correct += (logits.argmax(dim=1) == test_batch_labels).sum()
                teacher_robust_correct += (
                    teacher_logits.argmax(dim=1) == test_batch_labels
                ).sum()
                nat_teacher_robust_correct += (
                    nat_teacher_logits.argmax(dim=1) == test_batch_labels
                ).sum()

            test_adv = (student_robust_correct.float() / robust_total).item()
            teacher_test_acc = (teacher_robust_correct.float() / robust_total).item()
            nat_teacher_test_acc = (nat_teacher_robust_correct.float() / robust_total).item()

            logger.info(
                "student robust acc {:.4f}, teacher robust acc {:.4f}, "
                "nat teacher robust acc {:.4f}",
                test_adv,
                teacher_test_acc,
                nat_teacher_test_acc,
            )

            natural_total = 0
            student_natural_correct = torch.zeros((), device=device, dtype=torch.long)
            teacher_natural_correct = torch.zeros((), device=device, dtype=torch.long)
            nat_teacher_natural_correct = torch.zeros((), device=device, dtype=torch.long)

            for test_batch_data, test_batch_labels in iterate_device_batches(
                testloader, device, non_blocking
            ):
                with torch.inference_mode(), autocast_context(device, amp_dtype):
                    logits = student_eval(test_batch_data)
                    teacher_logits = teacher_eval(test_batch_data)
                    nat_teacher_logits = teacher_nat_eval(test_batch_data)
                natural_total += test_batch_labels.size(0)
                student_natural_correct += (logits.argmax(dim=1) == test_batch_labels).sum()
                teacher_natural_correct += (
                    teacher_logits.argmax(dim=1) == test_batch_labels
                ).sum()
                nat_teacher_natural_correct += (
                    nat_teacher_logits.argmax(dim=1) == test_batch_labels
                ).sum()

            test_nat = (student_natural_correct.float() / natural_total).item()
            teacher_test_accs_natural = (teacher_natural_correct.float() / natural_total).item()
            nat_teacher_test_accs_natural = (
                nat_teacher_natural_correct.float() / natural_total
            ).item()

            if epoch % 50 == 0:
                state = {
                    "model": unwrap_model(student).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                }
                torch.save(state, os.path.join(MODEL_DIR, f"student_{epoch}.pth"))
                state = {
                    "model": unwrap_model(teacher).state_dict(),
                    "optimizer": adv_teacher_optimizer.state_dict(),
                    "epoch": epoch,
                }
                torch.save(state, os.path.join(MODEL_DIR, f"teacher_{epoch}.pth"))

            if epoch > latest_checkpoint_start:
                state = {
                    "model": unwrap_model(student).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                }
                torch.save(state, os.path.join(MODEL_DIR, "student_latest.pth"))
                state = {
                    "model": unwrap_model(teacher).state_dict(),
                    "optimizer": adv_teacher_optimizer.state_dict(),
                    "epoch": epoch,
                }
                torch.save(state, os.path.join(MODEL_DIR, "teacher_latest.pth"))

            if (test_nat + test_adv) / 2 > best_accuracy:
                best_accuracy = (test_nat + test_adv) / 2
                state = {
                    "model": unwrap_model(student).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                }
                torch.save(state, os.path.join(MODEL_DIR, "student_best.pth"))
                state = {
                    "model": unwrap_model(teacher).state_dict(),
                    "optimizer": adv_teacher_optimizer.state_dict(),
                    "epoch": epoch,
                }
                torch.save(state, os.path.join(MODEL_DIR, "teacher_best.pth"))
                logger.info("best accuracy:{}", best_accuracy)

            logger.info(
                "student natural acc {:.4f}, adv teacher natural acc {:.4f}, "
                "nat teacher natural acc {:.4f}",
                test_nat,
                teacher_test_accs_natural,
                nat_teacher_test_accs_natural,
            )

            with open(os.path.join(MODEL_DIR, DRAW_FILE), "a", encoding="utf-8") as handle:
                handle.write(
                    f"{epoch} {test_adv} {test_nat} {teacher_test_acc} "
                    f"{teacher_test_accs_natural} {nat_teacher_test_acc} "
                    f"{nat_teacher_test_accs_natural}\n"
                )

    for monitor in feature_monitors.values():
        monitor.close()


if __name__ == "__main__":
    main()
