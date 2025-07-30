import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import parameters_to_vector
from typing import List
from visualizations import generate_visuals_sample
from models import FCNN, CNN
from utilities import compute_hvp, lanczos
from preprocessing import load_mnist, load_cifar10, load_cifar100
from loss_functions import MSELoss
from pathlib import Path

def runner_fcnn(
    dataset: str,
    subset_size: int,
    seed_data: int,
    seed_params: int,
    init_width: float,
    hidden_sizes: List[int],
    activation: str,
    loss_fn: str,
    learning_rate: float,
    iterations: int,
    hessian_freq: int,
    hessian_k: int
):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Using device: {device}")

    initial_state = torch.get_rng_state()

    torch.manual_seed(seed_data)

    if dataset == 'mnist':
        full_train, _ = load_mnist(loss_fn)
        input_dim = 784
        num_classes = 10
    elif dataset == 'cifar10':
        full_train, _ = load_cifar10(loss_fn)
        input_dim = 3072
        num_classes = 10
    elif dataset == 'cifar100':
        full_train, _ = load_cifar100(loss_fn)
        input_dim = 3072
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
        
    subset = Subset(full_train, torch.randperm(len(full_train))[:subset_size].tolist())
    loader = DataLoader(subset, batch_size = len(subset), shuffle = False)

    torch.set_rng_state(initial_state)
    torch.manual_seed(seed_params)

    model = FCNN(activation, hidden_sizes, input_dim, num_classes, init_width = init_width).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    criterion = {'cross_entropy': nn.CrossEntropyLoss(reduction = 'sum'), 'mse': MSELoss()}[loss_fn]
    criterion_sum = criterion

    prev_params = parameters_to_vector(model.parameters()).detach().clone().to(device)
    prev_param_dict = {name: param.clone().detach() for name, param in model.named_parameters()}
    cumulative_changes = {name: 0.0 for name, _ in model.named_parameters()}

    losses, accs, eigs, traj_lengths, grad_norms = [], [], [], [], []

    for epoch in range(1, iterations + 1):

        if epoch % hessian_freq == 0:
            hv_loader = DataLoader(subset, batch_size = len(subset), shuffle = False)
            current_params = parameters_to_vector(model.parameters()).detach().clone().to(device)
            evals = lanczos(
                lambda v: compute_hvp(model, criterion_sum, hv_loader, v),
                dim = len(current_params),
                neigs = hessian_k,
                device = device
            )
            eigs.append(evals.cpu().tolist())
        else:
            eigs.append([None] * hessian_k)

        output = None
        target = None
        loss = None
        acc = 0.0

        for data, target in loader:

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)

            target_tensor = target

            loss = criterion(output, target_tensor) / len(loader.dataset)
            loss.backward()
            
            # Alternative if we get memory issues, current is faster.
            # param_norms = torch.stack([p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None])
            # norm = param_norms.norm(2).item()

            grads = [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
            grad_norms.append(torch.cat(grads).norm().item())
            
            optimizer.step()

        if output is not None and target is not None and loss is not None:
            pred = output.argmax(dim = 1)
            if loss_fn == 'mse':
                true_labels = target.argmax(dim = 1)
                acc = pred.eq(true_labels).sum().item() / len(true_labels)
            else:
                acc = pred.eq(target).sum().item() / len(target)
            losses.append(loss.item())
            accs.append(acc)
        
        current_params = parameters_to_vector(model.parameters()).detach().clone().to(device)
        step = current_params - prev_params
        traj_lengths.append(step.norm().item())
        prev_params = current_params

        for name, param in model.named_parameters():
            delta = param.detach() - prev_param_dict[name]
            cumulative_changes[name] += delta.abs().norm().item()
            prev_param_dict[name] = param.detach().clone()


        print(f"Iter {epoch}/{iterations} Loss: {loss.item() if loss is not None else 'N/A'} Acc: {acc:.4f} Eigs: {eigs[-1]}")

    generate_visuals_sample(iterations, losses, accs, traj_lengths, grad_norms, eigs, cumulative_changes, learning_rate, hessian_k)

def reduced_runner_fcnn(
    dataset: str,
    subset_size: int,
    seed_data: int,
    seed_params: int,
    init_width: float,
    hidden_sizes: List[int],
    activation: str,
    loss_fn: str,
    learning_rate: float,
    iterations: int,
    initial_eig_threshold: int,
    initial_hessian_freq: int,
    final_hessian_freq: int,
    hessian_k: int,
    save_individual: bool = True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    initial_state = torch.get_rng_state()
    
    torch.manual_seed(seed_data)

    if dataset == 'mnist':
        full_train, _ = load_mnist(loss_fn)
        input_dim = 784
        num_classes = 10
    elif dataset == 'cifar10':
        full_train, _ = load_cifar10(loss_fn)
        input_dim = 3072
        num_classes = 10
    elif dataset == 'cifar100':
        full_train, _ = load_cifar100(loss_fn)
        input_dim = 3072
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
        
    subset = Subset(full_train, torch.randperm(len(full_train))[:subset_size].tolist())
    loader = DataLoader(subset, batch_size = len(subset), shuffle = False)
    
    torch.set_rng_state(initial_state)
    torch.manual_seed(seed_params)
    
    model = FCNN(activation, hidden_sizes, input_dim, num_classes, init_width = init_width).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    criterion = {'cross_entropy': nn.CrossEntropyLoss(reduction='sum'), 'mse': MSELoss()}[loss_fn]
    criterion_sum = criterion
    
    losses, accs, eigs = [], [], []
    
    log_file = None
    individual_dir = None
    
    if save_individual:
        num_layers = len(hidden_sizes) * 2 + 1
        individual_dir = Path("results") / "initializations" / "fcnn" / dataset / f"{activation}_{num_layers}_{loss_fn}_{iterations}_{seed_data}_{subset_size}" / str(learning_rate) / str(seed_params)
        individual_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = individual_dir / "training_log.csv"
        with open(log_file, 'w') as f:
            header = "Iteration, Loss, Accuracy"
            for i in range(hessian_k):
                header += f",Eigenvalue_{i+1}"
            f.write(header + "\n")

    for epoch in range(1, iterations + 1):

        current_hessian_freq = initial_hessian_freq if epoch <= initial_eig_threshold else final_hessian_freq
        
        if epoch % current_hessian_freq == 0:
            hv_loader = DataLoader(subset, batch_size = len(subset), shuffle = False)
            evals = lanczos(
                lambda v: compute_hvp(model, criterion_sum, hv_loader, v),
                dim = len(parameters_to_vector(model.parameters())),
                neigs = hessian_k,
                device = device
            )
            eigs.append(evals.cpu().tolist())
        else:
            eigs.append([None] * hessian_k)

        output = None
        target = None
        loss = None
        acc = 0.0

        for data, target in loader:
            
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)

            target_tensor = target

            loss = criterion(output, target_tensor) / len(loader.dataset)
            loss.backward()
            optimizer.step()

        if output is not None and target is not None and loss is not None:
            pred = output.argmax(dim = 1)
            if loss_fn == 'mse':
                true_labels = target.argmax(dim = 1)
                acc = pred.eq(true_labels).sum().item() / len(true_labels)
            else:
                acc = pred.eq(target).sum().item() / len(target)
            losses.append(loss.item())
            accs.append(acc)

        if save_individual and log_file is not None:
            with open(log_file, 'a') as f:
                row = f"{epoch},{loss.item() if loss is not None else 'N/A':.6f},{acc if output is not None else 'N/A':.6f}"
                for i in range(hessian_k):
                    eig_val = eigs[-1][i] if eigs[-1][i] is not None else ""
                    row += f",{eig_val}"
                f.write(row + "\n")

    print(f"Final Loss: {losses[-1] if losses else 'N/A'} Final Acc: {accs[-1] if accs else 'N/A'} Eigs: {eigs[-1] if eigs else 'N/A'}")    

    if save_individual and individual_dir is not None:
        model_path = individual_dir / "model_state_dict.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Saved model and logs to {individual_dir}")

    return {
        'losses': losses,
        'accs': accs,
        'eigs': eigs,
    }

def runner_cnn(
    dataset: str,
    widths: List[int],
    fc_sizes: List[int],
    subset_size: int,
    seed_data: int,
    seed_params: int,
    init_width: float,
    activation: str,
    loss_fn: str,
    learning_rate: float,
    iterations: int,
    hessian_freq: int,
    hessian_k: int
):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Using device: {device}")

    initial_state = torch.get_rng_state()

    torch.manual_seed(seed_data)

    if dataset == 'mnist':
        full_train, _ = load_mnist(loss_fn)
        input_dim = 784
        num_classes = 10
    elif dataset == 'cifar10':
        full_train, _ = load_cifar10(loss_fn)
        input_dim = 3072
        num_classes = 10
    elif dataset == 'cifar100':
        full_train, _ = load_cifar100(loss_fn)
        input_dim = 3072
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
        
    subset = Subset(full_train, torch.randperm(len(full_train))[:subset_size].tolist())
    loader = DataLoader(subset, batch_size = len(subset), shuffle = False)

    torch.set_rng_state(initial_state)
    torch.manual_seed(seed_params)

    model = CNN(dataset, widths, fc_sizes, activation, init_width).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    criterion = {'cross_entropy': nn.CrossEntropyLoss(reduction='sum'), 'mse': MSELoss()}[loss_fn]
    criterion_sum = criterion

    prev_params = parameters_to_vector(model.parameters()).detach().clone().to(device)
    prev_param_dict = {name: param.clone().detach() for name, param in model.named_parameters()}
    cumulative_changes = {name: 0.0 for name, _ in model.named_parameters()}

    losses, accs, eigs, traj_lengths, grad_norms = [], [], [], [], []

    for epoch in range(1, iterations + 1):

        if epoch % hessian_freq == 0:
            hv_loader = DataLoader(subset, batch_size = len(subset), shuffle = False)
            current_params = parameters_to_vector(model.parameters()).detach().clone().to(device)
            evals = lanczos(
                lambda v: compute_hvp(model, criterion_sum, hv_loader, v),
                dim = len(current_params),
                neigs = hessian_k,
                device = device
            )
            eigs.append(evals.cpu().tolist())
        else:
            eigs.append([None] * hessian_k)

        output = None
        target = None
        loss = None
        acc = 0.0

        for data, target in loader:

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)

            target_tensor = target

            loss = criterion(output, target_tensor) / len(loader.dataset)
            loss.backward()
            
            grads = [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
            grad_norms.append(torch.cat(grads).norm().item())
            
            optimizer.step()

        if output is not None and target is not None and loss is not None:
            pred = output.argmax(dim = 1)
            if loss_fn == 'mse':
                true_labels = target.argmax(dim = 1)
                acc = pred.eq(true_labels).sum().item() / len(true_labels)
            else:
                acc = pred.eq(target).sum().item() / len(target)
            losses.append(loss.item())
            accs.append(acc)
        
        current_params = parameters_to_vector(model.parameters()).detach().clone().to(device)
        step = current_params - prev_params
        traj_lengths.append(step.norm().item())
        prev_params = current_params

        for name, param in model.named_parameters():
            delta = param.detach() - prev_param_dict[name]
            cumulative_changes[name] += delta.abs().norm().item()
            prev_param_dict[name] = param.detach().clone()


        print(f"Iter {epoch}/{iterations} Loss: {loss.item() if loss is not None else 'N/A'} Acc: {acc:.4f} Eigs: {eigs[-1]}")

    generate_visuals_sample(iterations, losses, accs, traj_lengths, grad_norms, eigs, cumulative_changes, learning_rate, hessian_k)