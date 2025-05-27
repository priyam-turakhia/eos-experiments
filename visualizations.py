import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt

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