import tensorflow as tf
from tensorflow.keras import Model, layers


class TeacherPINN(Model):
    def __init__(self, input_dim=8, hidden_layers=None, output_dim=4):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        hidden_layers = [256] * 6 if hidden_layers is None else list(hidden_layers)
        if any(not isinstance(width, int) or width <= 0 for width in hidden_layers):
            raise ValueError("hidden layer widths must be positive integers")

        self.input_layer = layers.InputLayer(shape=(input_dim,))
        self.hidden_layers = [
            layers.Dense(
                neurons,
                activation="swish",
                kernel_initializer="glorot_uniform",
            )
            for neurons in hidden_layers
        ]
        self.output_layer = layers.Dense(output_dim, activation="linear")

    def call(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)


class StudentPINN(Model):
    def __init__(self, input_dim=8, hidden_layers=None, output_dim=4):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        hidden_layers = [64] * 3 if hidden_layers is None else list(hidden_layers)
        if any(not isinstance(width, int) or width <= 0 for width in hidden_layers):
            raise ValueError("hidden layer widths must be positive integers")

        self.input_layer = layers.InputLayer(shape=(input_dim,))
        self.hidden_layers = [
            layers.Dense(
                neurons,
                activation="swish",
                kernel_initializer="glorot_uniform",
            )
            for neurons in hidden_layers
        ]
        self.output_layer = layers.Dense(output_dim, activation="linear")

    def call(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)
