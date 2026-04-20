import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from contextlib import nullcontext


def _autocast_context(device, amp_enabled=False, amp_dtype=None):
    if not amp_enabled or amp_dtype is None or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def _unwrap_model(model):
    return getattr(model, "_orig_mod", model)


def _set_model_training(model, training):
    eager_model = _unwrap_model(model)
    model.train(training)
    if eager_model is not model and eager_model.training != training:
        eager_model.train(training)
    return eager_model


def attack_pgd(model,train_batch_data,train_batch_labels,attack_iters=10,step_size=2/255.0,epsilon=8.0/255.0):
    eager_model = _unwrap_model(model)
    device = next(eager_model.parameters()).device
    ce_loss = torch.nn.CrossEntropyLoss().to(device)
    train_ifgsm_data = train_batch_data.detach() + torch.zeros_like(train_batch_data).uniform_(-epsilon,epsilon)
    train_ifgsm_data = torch.clamp(train_ifgsm_data,0,1)
    was_training = eager_model.training
    _set_model_training(model, False)
    for i in range(attack_iters):
        train_ifgsm_data.requires_grad_()
        with torch.enable_grad():
            logits = eager_model(train_ifgsm_data)
            loss = ce_loss(logits,train_batch_labels.to(device))
        train_grad = torch.autograd.grad(loss, [train_ifgsm_data])[0].detach()
        train_ifgsm_data = train_ifgsm_data + step_size*torch.sign(train_grad)
        train_ifgsm_data = torch.clamp(train_ifgsm_data.detach(),0,1)
        train_ifgsm_pert = train_ifgsm_data - train_batch_data
        train_ifgsm_pert = torch.clamp(train_ifgsm_pert,-epsilon,epsilon)
        train_ifgsm_data = train_batch_data + train_ifgsm_pert
        train_ifgsm_data = train_ifgsm_data.detach()
    _set_model_training(model, was_training)
    return train_ifgsm_data

def robust_inner_loss_push(model,
                teacher_adv_model,
                teacher_nat,
                x_natural,
                y,
                optimizer,
                teacher_adv_optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=6.0,
                amp_enabled=False,
                amp_dtype=None):

    device = x_natural.device
    eager_model = _unwrap_model(model)
    criterion_ce_loss = torch.nn.CrossEntropyLoss().to(device)
    _set_model_training(model, False)
    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural).detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            with torch.autocast(device_type=device.type, enabled=False):
                loss_ce = criterion_ce_loss(eager_model(x_adv), y.to(device))
        grad = torch.autograd.grad(loss_ce, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    _set_model_training(model, True)
    _set_model_training(teacher_adv_model, True)
    _set_model_training(teacher_nat, False)
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    optimizer.zero_grad(set_to_none=True)
    teacher_adv_optimizer.zero_grad(set_to_none=True)
    
    with _autocast_context(device, amp_enabled=amp_enabled, amp_dtype=amp_dtype):
        student_logits = model(x_adv)
        teacher_logits = teacher_adv_model(x_adv)
        nat_logits = teacher_nat(x_adv)

    return student_logits, teacher_logits, nat_logits, x_adv

def CIARD_inner_loss(model,
                teacher_adv_model,
                teacher_nat,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=6.0):

    device = x_natural.device
    criterion_ce_loss = torch.nn.CrossEntropyLoss().to(device)
    model.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural).detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_ce = criterion_ce_loss(model(x_adv), y.to(device))
        grad = torch.autograd.grad(loss_ce, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    optimizer.zero_grad()
    student_logits = model(x_adv)
    teacher_adv_model.eval()
    teacher_nat.eval()
    with torch.no_grad():
        teacher_logits = teacher_adv_model(x_adv)
        nat_logits = teacher_nat(x_adv)
    return student_logits, teacher_logits, nat_logits, x_adv
