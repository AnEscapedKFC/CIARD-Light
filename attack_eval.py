import torch
from cifar10_models import *
from cifar10_nat_teacher_models import *
import torchvision
from torchvision import transforms
from loguru import logger
import numpy as np
import torchattacks
import torch.nn.functional as F

from autoattack import AutoAttack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError(
        "attack_eval.py requires an NVIDIA CUDA GPU. For RTX 5070 Ti, follow setup_rtx5070ti.md."
    )


def load_checkpoint(path):
    load_kwargs = {"map_location": torch.device("cpu")}
    try:
        return torch.load(path, weights_only=True, **load_kwargs)
    except TypeError:
        return torch.load(path, **load_kwargs)


def load_state_dict_from_checkpoint(path, key="model"):
    checkpoint = load_checkpoint(path)
    if isinstance(checkpoint, dict) and key in checkpoint and hasattr(checkpoint[key], "items"):
        checkpoint = checkpoint[key]
    if not hasattr(checkpoint, "items"):
        raise TypeError(f"Unsupported checkpoint format in {path}")
    return {k.replace('module.', ''): v for k, v in checkpoint.items()}


def eval_autoattack(model, testloader, epsilon=8/255.0, norm='Linf', attacks_to_run=None):
    model.eval()
    adversary = AutoAttack(model, norm=norm, eps=epsilon, version='standard', verbose=True)
    if attacks_to_run is not None:
        adversary.attacks_to_run = attacks_to_run  # e.g., ['apgd-ce', 'apgd-dlr', 'fab', 'square']

    
    xs, ys = [], []
    for x, y in testloader:
        xs.append(x)
        ys.append(y)
    x_test = torch.cat(xs, dim=0).to(device)
    y_test = torch.cat(ys, dim=0).to(device)

    return adversary.run_standard_evaluation(x_test, y_test, bs=min(128, x_test.size(0)))

path = ""
if not path:
    raise ValueError("Set `path` in attack_eval.py to the student checkpoint before running evaluation.")

student = mobilenet_v2()# cifar10_resnet56()# wideresnet()##resnet18()#

teacher1_path =  'models/model_cifar_wrn.pt' #for blackbox attack
teacher = wideresnet()

transform_test = transforms.Compose([
    transforms.ToTensor(),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

student.load_state_dict(load_state_dict_from_checkpoint(path))
student = student.to(device)
student.eval()



def attack_pgd(model,train_batch_data,train_batch_labels,attack_iters=10,step_size=2/255.0,epsilon=8.0/255.0):
    device = next(model.parameters()).device
    ce_loss = torch.nn.CrossEntropyLoss().to(device)
    train_ifgsm_data = train_batch_data.detach() + torch.zeros_like(train_batch_data).uniform_(-epsilon,epsilon)
    train_ifgsm_data = torch.clamp(train_ifgsm_data,0,1)
    for i in range(attack_iters):
        train_ifgsm_data.requires_grad_()
        logits = model(train_ifgsm_data)
        loss = ce_loss(logits,train_batch_labels.to(device))
        train_grad = torch.autograd.grad(loss, train_ifgsm_data)[0].detach()
        train_ifgsm_data = train_ifgsm_data + step_size*torch.sign(train_grad)
        train_ifgsm_data = torch.clamp(train_ifgsm_data.detach(),0,1)
        train_ifgsm_pert = train_ifgsm_data - train_batch_data
        train_ifgsm_pert = torch.clamp(train_ifgsm_pert,-epsilon,epsilon)
        train_ifgsm_data = train_batch_data + train_ifgsm_pert
        train_ifgsm_data = train_ifgsm_data.detach()
    return train_ifgsm_data

def attack_fgsm(model, train_batch_data, train_batch_labels, epsilon=8.0/255.0):
    device = next(model.parameters()).device
    ce_loss = torch.nn.CrossEntropyLoss().to(device)

    train_batch_data.requires_grad_()
    logits = model(train_batch_data)
    loss = ce_loss(logits, train_batch_labels.to(device))
    data_grad = torch.autograd.grad(loss, train_batch_data)[0].detach()
    sign_data_grad = data_grad.sign()
    
    perturbed_data = train_batch_data + epsilon * sign_data_grad
    perturbed_data = torch.clamp(perturbed_data, 0, 1) 
    return perturbed_data

def attack_cw_inf(model, inputs, target, confidence=50, num_classes=10, epsilon=8/255, lr=2/255, steps=30):
    perturbation = torch.zeros_like(inputs).to(inputs.device).requires_grad_()
    for _ in range(steps):
        output = model(inputs + perturbation)
        target_onehot = F.one_hot(target, num_classes=num_classes).float().to(inputs.device)
        real = torch.sum(target_onehot * output, dim=1)
        other = torch.max((1 - target_onehot) * output - target_onehot * 10000, dim=1)[0]
        loss = -torch.clamp(real - other + confidence, min=0.).mean()  
        grad = torch.autograd.grad(loss, perturbation)[0]
        perturbation = (perturbation + lr * torch.sign(grad)).clamp(-epsilon, epsilon)
        perturbation = perturbation.detach().requires_grad_()
    adversarial_input = inputs + perturbation
    adversarial_input = torch.clamp(adversarial_input, 0, 1) 
    return adversarial_input 
logger.info("=============== AutoAttack Evaluation ===============")
eval_autoattack(student, testloader, epsilon=8/255.0, norm='Linf')

logger.info("============white box attack===================")

torch.cuda.empty_cache()
test_accs_naturals = []
for step,(test_batch_data,test_batch_labels) in enumerate(testloader): #,index
    test_batch_data = test_batch_data.float().to(device)
    test_batch_labels = test_batch_labels.to(device)
    with torch.no_grad():
        logits = student(test_batch_data)
    predictions = np.argmax(logits.cpu().detach().numpy(),axis=1)
    predictions = predictions - test_batch_labels.cpu().detach().numpy()
    test_accs_naturals = test_accs_naturals + predictions.tolist()
test_accs_naturals = np.array(test_accs_naturals)
test_nat = np.sum(test_accs_naturals==0)/len(test_accs_naturals)
text = f'student clean acc:  {test_nat:.4f}'
logger.info(text)

torch.cuda.empty_cache()
test_accs = []
for step,(test_batch_data,test_batch_labels) in enumerate(testloader):#,index
    test_batch_data = test_batch_data.float().to(device)
    test_batch_labels = test_batch_labels.to(device)
    test_ifgsm_data = attack_pgd(student,test_batch_data,test_batch_labels,attack_iters=20,step_size=0.003,epsilon=8.0/255.0)
    with torch.no_grad():
        logits = student(test_ifgsm_data)

    predictions = np.argmax(logits.cpu().detach().numpy(),axis=1)
    predictions = predictions - test_batch_labels.cpu().detach().numpy()
    test_accs = test_accs + predictions.tolist()
test_accs = np.array(test_accs)
test_adv = np.sum(test_accs==0)/len(test_accs)
text = f'student robust acc under PGD_trades Attack {test_adv:.4f}'
logger.info(text)

torch.cuda.empty_cache()
test_accs = []
for step,(test_batch_data,test_batch_labels) in enumerate(testloader):#,index
    test_batch_data = test_batch_data.float().to(device)
    test_batch_labels = test_batch_labels.to(device)
    test_ifgsm_data = attack_pgd(student,test_batch_data,test_batch_labels,attack_iters=20,step_size=2.0/255.0,epsilon=8.0/255.0)
    with torch.no_grad():
        logits = student(test_ifgsm_data)

    predictions = np.argmax(logits.cpu().detach().numpy(),axis=1)
    predictions = predictions - test_batch_labels.cpu().detach().numpy()
    test_accs = test_accs + predictions.tolist()
test_accs = np.array(test_accs)
test_adv = np.sum(test_accs==0)/len(test_accs)
text = f'student robust acc under PGD_sat Attack {test_adv:.4f}'
logger.info(text)

torch.cuda.empty_cache()
test_accs = []
for step,(test_batch_data,test_batch_labels) in enumerate(testloader):#,index
    test_batch_data = test_batch_data.float().to(device)
    test_batch_labels = test_batch_labels.to(device)
    test_ifgsm_data = attack_fgsm(student,test_batch_data,test_batch_labels)
    with torch.no_grad():
        logits = student(test_ifgsm_data)

    predictions = np.argmax(logits.cpu().detach().numpy(),axis=1)
    predictions = predictions - test_batch_labels.cpu().detach().numpy()
    test_accs = test_accs + predictions.tolist()
test_accs = np.array(test_accs)
test_adv = np.sum(test_accs==0)/len(test_accs)
text = f'student robust acc under FGSM Attack {test_adv:.4f}'
logger.info(text)

torch.cuda.empty_cache()
test_accs = []
for step,(test_batch_data,test_batch_labels) in enumerate(testloader):#,index
    test_batch_data = test_batch_data.float().to(device)
    test_batch_labels = test_batch_labels.to(device)
    test_ifgsm_data = attack_cw_inf(student,test_batch_data,test_batch_labels)
    with torch.no_grad():
        logits = student(test_ifgsm_data)

    predictions = np.argmax(logits.cpu().detach().numpy(),axis=1)
    predictions = predictions - test_batch_labels.cpu().detach().numpy()
    test_accs = test_accs + predictions.tolist()
test_accs = np.array(test_accs)
test_adv = np.sum(test_accs==0)/len(test_accs)
text = f'student robust acc under CW L_inf {test_adv:.4f}'
logger.info(text)




teacher.load_state_dict(load_state_dict_from_checkpoint(teacher1_path, key="model"))
teacher = teacher.to(device)
teacher.eval()
logger.info("===============blackbox attack================")

torch.cuda.empty_cache()
test_accs = []
for step,(test_batch_data,test_batch_labels) in enumerate(testloader):#,index
    test_batch_data = test_batch_data.float().to(device)
    test_batch_labels = test_batch_labels.to(device)
    test_ifgsm_data = attack_pgd(teacher,test_batch_data,test_batch_labels,attack_iters=20,step_size=0.003,epsilon=8.0/255.0)
    with torch.no_grad():
        logits = student(test_ifgsm_data)

    predictions = np.argmax(logits.cpu().detach().numpy(),axis=1)
    predictions = predictions - test_batch_labels.cpu().detach().numpy()
    test_accs = test_accs + predictions.tolist()
test_accs = np.array(test_accs)
test_adv = np.sum(test_accs==0)/len(test_accs)
text = f'student robust acc under PGD_trades Attack {test_adv:.4f}'
logger.info(text)

torch.cuda.empty_cache()
test_accs = []
attack_sa = torchattacks.Square(student, norm='Linf', eps=8/255, n_queries=100)
for step,(test_batch_data,test_batch_labels) in enumerate(testloader):#,index
    test_batch_data = test_batch_data.float().to(device)
    test_batch_labels = test_batch_labels.to(device)
    
    
    test_ifgsm_data = attack_sa(test_batch_data,test_batch_labels)
    with torch.no_grad():
        logits = student(test_ifgsm_data)

    predictions = np.argmax(logits.cpu().detach().numpy(),axis=1)
    predictions = predictions - test_batch_labels.cpu().detach().numpy()
    test_accs = test_accs + predictions.tolist()
test_accs = np.array(test_accs)
test_adv = np.sum(test_accs==0)/len(test_accs)
text = f'student robust acc under Square Attack {test_adv:.4f}'
logger.info(text)

torch.cuda.empty_cache()
test_accs = []
for step,(test_batch_data,test_batch_labels) in enumerate(testloader):#,index
    test_batch_data = test_batch_data.float().to(device)
    test_batch_labels = test_batch_labels.to(device)
    test_ifgsm_data = attack_cw_inf(teacher,test_batch_data,test_batch_labels)
    with torch.no_grad():
        logits = student(test_ifgsm_data)

    predictions = np.argmax(logits.cpu().detach().numpy(),axis=1)
    predictions = predictions - test_batch_labels.cpu().detach().numpy()
    test_accs = test_accs + predictions.tolist()
test_accs = np.array(test_accs)
test_adv = np.sum(test_accs==0)/len(test_accs)
text = f'student robust acc under CW L_inf {test_adv:.4f}'
logger.info(text)






