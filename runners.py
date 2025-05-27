import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import parameters_to_vector
from torchvision import datasets, transforms
from typing import List
from visualizations import generate_visuals_sample
from models import FCNN
from utilities import compute_hvp, lanczos
from pathlib import Path

def runner_fcnn_mnist(
    subset_size: int,
    seed_data: int,
    seed_params: int,
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
    full_train = datasets.MNIST(root = './data', train = True, download = True, transform = transforms.ToTensor())
    subset = Subset(full_train, torch.randperm(len(full_train))[:subset_size].tolist())
    loader = DataLoader(subset, batch_size = len(subset), shuffle = False)

    torch.set_rng_state(initial_state)
    torch.manual_seed(seed_params)

    model = FCNN(activation, hidden_sizes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    criterion = {'cross_entropy': nn.CrossEntropyLoss(), 'mse': nn.MSELoss()}[loss_fn]
    criterion_sum = {'cross_entropy': nn.CrossEntropyLoss(reduction = 'sum'), 'mse': nn.MSELoss(reduction = 'sum')}[loss_fn]

    prev_params = parameters_to_vector(model.parameters()).detach().clone().to(device)
    prev_param_dict = {name: param.clone().detach() for name, param in model.named_parameters()}
    cumulative_changes = {name: 0.0 for name, _ in model.named_parameters()}

    losses, accs, eigs, traj_lengths = [], [], [], []

    for epoch in range(1, iterations + 1):

        for data, target in loader:

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)

            if loss_fn == 'mse':
                target_tensor = F.one_hot(target, num_classes=output.size(1)).float().to(device)
            else:
                target_tensor = target

            loss = criterion(output, target_tensor)
            loss.backward()
            optimizer.step()

        pred = output.argmax(dim = 1)
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

        if epoch % hessian_freq == 0:
            hv_loader = DataLoader(subset, batch_size = len(subset), shuffle = False)
            evals = lanczos(
                lambda v: compute_hvp(model, criterion_sum, hv_loader, v),
                dim = len(current_params),
                neigs = hessian_k,
                device = device
            )
            eigs.append(evals.cpu().tolist())
        else:
            eigs.append([None] * hessian_k)

        print(f"Iter {epoch}/{iterations} Loss: {loss.item():.4f} Acc: {acc:.4f} Eigs: {eigs[-1]}")

    generate_visuals_sample(iterations, losses, accs, traj_lengths, eigs, cumulative_changes, learning_rate, hessian_k)

def reduced_runner_init_fcnn_mnist(
    subset_size: int,
    seed_data: int,
    seed_params: int,
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
    full_train = datasets.MNIST(root = './data', train = True, download = True, transform = transforms.ToTensor())
    subset = Subset(full_train, torch.randperm(len(full_train))[:subset_size].tolist())
    loader = DataLoader(subset, batch_size = len(subset), shuffle = False)
    
    torch.set_rng_state(initial_state)
    torch.manual_seed(seed_params)
    
    model = FCNN(activation, hidden_sizes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    criterion = {'cross_entropy': nn.CrossEntropyLoss(), 'mse': nn.MSELoss()}[loss_fn]
    criterion_sum = {'cross_entropy': nn.CrossEntropyLoss(reduction = 'sum'), 'mse': nn.MSELoss(reduction = 'sum')}[loss_fn]
    
    losses, accs, eigs = [], [], []
    
    if save_individual:
        
        num_layers = len(hidden_sizes) * 2 + 1
        individual_dir = Path("results") / "initializations" / "fcnn" / "mnist" / f"{activation}_{num_layers}_{loss_fn}_{iterations}_{seed_data}_{subset_size}" / str(learning_rate) / str(seed_params)
        individual_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = individual_dir / "training_log.csv"
        with open(log_file, 'w') as f:
            header = "Iteration, Loss, Accuracy"
            for i in range(hessian_k):
                header += f",Eigenvalue_{i+1}"
            f.write(header + "\n")

    for epoch in range(1, iterations + 1):

        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)

            if loss_fn == 'mse':
                target_tensor = F.one_hot(target, num_classes=output.size(1)).float().to(device)
            else:
                target_tensor = target

            loss = criterion(output, target_tensor)
            loss.backward()
            optimizer.step()

        pred = output.argmax(dim = 1)
        acc = pred.eq(target).sum().item() / len(target)
        losses.append(loss.item())
        accs.append(acc)

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

        if save_individual:
            with open(log_file, 'a') as f:
                row = f"{epoch},{loss.item():.6f},{acc:.6f}"
                for i in range(hessian_k):
                    eig_val = eigs[-1][i] if eigs[-1][i] is not None else ""
                    row += f",{eig_val}"
                f.write(row + "\n")

    print(f"Final Loss: {losses[-1]:.4f} Final Acc: {accs[-1]:.4f} Eigs: {eigs[-1]}")    

    if save_individual:
        model_path = individual_dir / "model_state_dict.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Saved model and logs to {individual_dir}")

    return {
        'losses': losses,
        'accs': accs,
        'eigs': eigs,
    }