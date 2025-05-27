import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
from pathlib import Path

def generate_visuals_sample(
    iterations: int,
    losses: List[float],
    accs: List[float],
    traj_lengths: List[float],
    eigs: List[List[float]],
    cumulative_changes: Dict[str, float],
    learning_rate: float,
    hessian_k: int
):
    iters = list(range(1, iterations + 1))
    
    min_loss, max_loss = min(losses), max(losses)
    padding = 0.1 * (max_loss - min_loss)
    plt.figure()
    plt.plot(iters, losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.ylim(min_loss - padding, max_loss + padding)
    plt.title('Training Loss')
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(iters, accs)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.title('Training Accuracy')
    plt.grid(True)
    plt.show()

    cum_dist = np.cumsum(traj_lengths) 
    plt.figure()
    plt.plot(iters, cum_dist)
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Distance')
    plt.title('Cumulative Optimization Trajectory Length')
    plt.grid(True)
    plt.show()

    plt.figure()
    for i in range(hessian_k):
        xs, ys = [], []
        for epoch, row in enumerate(eigs, start=1):
            val = row[i]
            if val is not None:
                xs.append(epoch)
                ys.append(val)
        plt.plot(xs, ys, label=f'λ_{i+1}')
    plt.axhline(2/learning_rate, linestyle=':', label='2/η')
    plt.xlabel('Iteration')
    plt.ylabel('Eigenvalue')
    plt.title('Top Hessian Eigenvalues')
    plt.legend()
    plt.grid(True)
    plt.show()

    top = sorted(cumulative_changes.items(), key=lambda x: x[1], reverse=True)[:5]
    names, vals = zip(*top)
    plt.figure()
    plt.barh(names, vals)
    plt.xlabel('Cumulative Parameter Change Norm')
    plt.title('Top 5 Principal Parameters')
    plt.grid(True)
    plt.show()
    print('Top 5 parameters by cumulative change:', names)

def generate_initialization_visualizations(
    all_trajectories: List[List[float]],
    all_accs: List[List[float]],
    learning_rate: float,
    iterations: int,
    initial_eig_threshold: int,
    initial_hessian_freq: int,
    final_hessian_freq: int,
    activation: str,
    hidden_sizes: List[int],
    loss_fn: str,
    seed_data: int,
    subset_size: int
):
    
    num_layers = len(hidden_sizes) * 2 + 1
    save_dir = Path("results") / "initializations" / "fcnn" / "mnist" / f"{activation}_{num_layers}_{loss_fn}_{iterations}_{seed_data}_{subset_size}" / str(learning_rate)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    trajectory_matrix = np.array(all_trajectories)

    x_values = []
    for epoch in range(1, iterations + 1):
        current_hessian_freq = initial_hessian_freq if epoch <= initial_eig_threshold else final_hessian_freq
        if epoch % current_hessian_freq == 0:
            x_values.append(epoch)

    x_values = np.array(x_values)

    plt.figure(figsize=(12, 7))

    for traj in all_trajectories:
        plt.plot(x_values, traj, color='lightgray', linewidth=0.5, alpha=0.1)

    percentiles = [0, 25, 50, 75, 100]
    percentile_curves = np.percentile(trajectory_matrix, percentiles, axis=0)

    labels = ['Minimum', '25th percentile', 'Median', '75th percentile', 'Maximum']
    colors = ['blue', 'green', 'black', 'orange', 'red']
    line_styles = ['--', '-.', '-', '-.', '--']

    for i, (curve, label, color, ls) in enumerate(zip(percentile_curves, labels, colors, line_styles)):
        plt.plot(x_values, curve, color=color, linewidth=2 if i == 2 else 1.5, 
                linestyle=ls, label=label)

    plt.fill_between(x_values, percentile_curves[1], percentile_curves[3], 
                    color='lightblue', alpha=0.3, label='Interquartile range')

    plt.axhline(2/learning_rate, linestyle=':', color='purple', linewidth=2, label='2/η')

    x_grid_ticks = np.arange(0, iterations + 1, 250)
    plt.xticks(x_grid_ticks)
    plt.grid(True, alpha=0.7)

    plt.xlabel('Iteration')
    plt.ylabel('Top Eigenvalue (Sharpness)')
    plt.title(f'Sharpness vs. Iteration (η = {learning_rate})')
    plt.legend()

    sharpness_filename = f"sharpness_lr_{learning_rate}.png"
    plt.savefig(save_dir / sharpness_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 7))
    
    for acc in all_accs:
        plt.plot(acc, color='lightgray', linewidth=0.5, alpha=0.1)
    
    acc_matrix = np.array(all_accs)
    acc_percentiles = [0, 25, 50, 75, 100]
    acc_percentile_curves = np.percentile(acc_matrix, acc_percentiles, axis=0)
    
    acc_labels = ['Minimum', '25th percentile', 'Median', '75th percentile', 'Maximum']
    acc_colors = ['blue', 'green', 'black', 'orange', 'red']
    acc_line_styles = ['--', '-.', '-', '-.', '--']
    
    for i, (curve, label, color, ls) in enumerate(zip(acc_percentile_curves, acc_labels, acc_colors, acc_line_styles)):
        plt.plot(curve, color=color, linewidth=2 if i == 2 else 1.5, 
                 linestyle=ls, label=label)
    
    plt.fill_between(range(len(all_accs[0])), acc_percentile_curves[1], acc_percentile_curves[3], 
                     color='lightblue', alpha=0.3, label='Interquartile range')
    
    acc_x_ticks = np.arange(0, len(all_accs[0]), 250)
    plt.xticks(acc_x_ticks)
    plt.grid(True, alpha=0.7)

    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.title(f'Accuracy vs. Iteration (η = {learning_rate})')
    plt.legend()
    
    accuracy_filename = f"accuracy_lr_{learning_rate}.png"
    plt.savefig(save_dir / accuracy_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualizations to {save_dir}")