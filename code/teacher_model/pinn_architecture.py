import tensorflow as tf
from tensorflow.keras import layers, Model

class TeacherPINN(Model):
    def __init__(self, input_dim=8, hidden_layers=[256]*6, output_dim=4):
        super(TeacherPINN, self).__init__()
        self.input_layer = layers.InputLayer(input_shape=(input_dim,))
        self.hidden_layers = []
        for neurons in hidden_layers:
            self.hidden_layers.append(
                layers.Dense(neurons, activation='swish',
                             kernel_initializer='glorot_uniform')
            )
        self.output_layer = layers.Dense(output_dim, activation='linear')
    
    def call(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

class StudentPINN(Model):
    def __init__(self, input_dim=8, hidden_layers=[64]*3, output_dim=4):
        super(StudentPINN, self).__init__()
        self.input_layer = layers.InputLayer(input_shape=(input_dim,))
        self.hidden_layers = []
        for neurons in hidden_layers:
            self.hidden_layers.append(
                layers.Dense(neurons, activation='swish',
                             kernel_initializer='glorot_uniform')
            )
        self.output_layer = layers.Dense(output_dim, activation='linear')
    
    def call(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)