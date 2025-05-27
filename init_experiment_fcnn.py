from multiprocessing import Pool
from runners import reduced_runner_init_fcnn_mnist
from visualizations import generate_initialization_visualizations

def run_single_lr_experiment(lr):

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
    num_samples = 5

    print(f"\n===== Running experiments for learning rate {lr} =====\n")
    
    all_trajectories = []
    all_accs = []
    
    for i in range(num_samples):

        seed_params = i
        
        print(f"Sample {i + 1}/{num_samples} (LR = {lr})")
        
        result = reduced_runner_init_fcnn_mnist(
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
            hessian_k = hessian_k,
            save_individual = True
        )
        
        top_eigs = []
        for epoch_eigs in result['eigs']:
            if epoch_eigs[0] is not None:
                top_eigs.append(epoch_eigs[0])
        
        all_trajectories.append(top_eigs)
        all_accs.append(result['accs'])
    
    generate_initialization_visualizations(
        all_trajectories = all_trajectories,
        all_accs = all_accs,
        learning_rate = lr,
        iterations = iterations,
        initial_eig_threshold = initial_eig_threshold,
        initial_hessian_freq = initial_hessian_freq,
        final_hessian_freq = final_hessian_freq,
        activation = activation,
        hidden_sizes = hidden_sizes,
        loss_fn = loss_fn,
        seed_data = seed_data,
        subset_size = subset_size
    )
    
    print(f"Completed learning rate {lr}")

if __name__ == "__main__":
    learning_rates = [0.05, 0.08]
    
    with Pool(processes = 2) as pool:
        pool.map(run_single_lr_experiment, learning_rates)
    
    print("All experiments completed!")