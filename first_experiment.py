import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from mnist_fcnn_utilities import reduced_run_experiment

subset_size = 5000
seed_data = 21
hidden_sizes = [32, 32]
activation = 'relu'
loss_fn = 'mse'
iterations = 5000
initial_eig_threshold = iterations - 1000
initial_hessian_freq = 1000
final_hessian_freq = 100
hessian_k = 3
num_samples = 20

results_dir = Path("results/first_experiment")
results_dir.mkdir(parents = True, exist_ok = True)

# learning_rates = [1e-4, 1e-3, 1e-2, 1e-1]
learning_rates = [0.01, 0.1]

for lr in learning_rates:

    print(f"\n===== Running experiments for learning rate {lr} =====\n")
    
    all_trajectories = []
    all_accs = []
    
    for i in range(num_samples):

        seed_params = i
        
        print(f"Sample {i + 1}/{num_samples} (LR = {lr})")
        
        result = reduced_run_experiment(
            subset_size = subset_size,
            seed_data = seed_data,
            seed_params = seed_params,
            hidden_sizes = hidden_sizes,
            activation = activation,
            loss_fn = loss_fn,
            learning_rate = lr,
            iterations = iterations,
            initial_eig_threshold = initial_eig_threshold,
            initial_hessian_freq = initial_hessian_freq,
            final_hessian_freq = final_hessian_freq,
            hessian_k = hessian_k
        )
        
        top_eigs = []
        for epoch_eigs in result['eigs']:
            if epoch_eigs[0] is not None:
                top_eigs.append(epoch_eigs[0])
        
        all_trajectories.append(top_eigs)
        all_accs.append(result['accs'])
    
    # Create sharpness visualization
    
    trajectory_matrix = np.array(all_trajectories)

    x_values = []
    current_index = 0
    for epoch in range(1, iterations + 1):
        current_hessian_freq = initial_hessian_freq if epoch <= initial_eig_threshold else final_hessian_freq
        if epoch % current_hessian_freq == 0:
            x_values.append(epoch)
            current_index += 1

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

    plt.axhline(2/lr, linestyle=':', color='purple', linewidth=2, label='2/η')

    x_grid_ticks = np.arange(0, iterations + 1, 250)
    plt.xticks(x_grid_ticks)
    plt.grid(True, alpha = 0.7)

    plt.xlabel('Iteration')
    plt.ylabel('Top Eigenvalue (Sharpness)')
    plt.title(f'Sharpness vs. Iteration (η = {lr})')
    plt.legend()

    filename = f"step_size_{str(lr)}_sharpness.png"
    plt.savefig(results_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create accuracy visualization
    
    plt.figure(figsize = (12, 7))
    
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
    
    acc_X_ticks = np.arange(0, len(all_accs[0]), 250)
    plt.grid(True, alpha = 0.7)

    plt.xticks(acc_X_ticks)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.title(f'Accuracy vs. Iteration (η = {lr})')
    plt.legend()
    
    filename = f"step_size_{str(lr)}_accuracy.png"
    plt.savefig(results_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()