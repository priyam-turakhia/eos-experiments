from runners import runner_fcnn, runner_cnn

# runner_fcnn(
#     dataset = 'cifar10',
#     subset_size = 5000,
#     seed_data = 1,
#     seed_params = 1,
#     init_width = 1,
#     hidden_sizes = [32, 32],
#     activation = "relu",
#     loss_fn = "mse",
#     learning_rate = 2/100,
#     iterations = 10000,
#     hessian_freq = 100,
#     hessian_k = 3
# )

runner_cnn(
    dataset = 'mnist',
    widths = [6, 16],
    fc_sizes = [120, 84],
    subset_size = 5000,
    seed_data = 1,
    seed_params = 1,
    init_width = 1,
    activation = "tanh",
    loss_fn = "mse",
    learning_rate = 0.02,
    iterations = 10000,
    hessian_freq = 100,
    hessian_k = 3   
)