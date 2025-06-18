from runners import runner_fcnn_mnist

runner_fcnn_mnist(
    10000,
    42,
    69,
    4.5,
    [32, 32],
    "relu",
    "mse",
    0.08,
    10000,
    500,
    3
),m