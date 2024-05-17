import numpy as np
import pandas as pd


class Neuronio:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoidDerivative(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        self.output = self.sigmoid(self.z)
        return self.output

    def back_propagation(self, values, learning_rate):
        self.dweights = np.dot(self.inputs.T, values * self.sigmoidDerivative(self.output))
        self.dbias = np.sum(values * self.sigmoidDerivative(self.output), axis=0)
        self.dinputs = np.dot(values * self.sigmoidDerivative(self.output), self.weights.T)

        self.weights += learning_rate * self.dweights
        self.bias += learning_rate * self.dbias

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.hidden_neuron = Neuronio(input_size, hidden_size)
        self.output_neuron = Neuronio(hidden_size, output_size)
        self.learning_rate = learning_rate

    def calculateAccuracy(self, targets, output_error):
        accuracy =  sum(output_error) / sum(targets)
        if accuracy >= 0:
            return accuracy
        else:
            return -accuracy

    def melhorEpoca(self, acuracias, indices):
        maior = acuracias[0]
        index = indices[0]
        for i in range(len(acuracias) - 1):
            if acuracias[i] >= maior:
                maior = acuracias[i]
                acuracias[i] = acuracias[i + 1]
                acuracias[i + 1] = maior

                index = indices[i]
                indices[i] = indices[i + 1]
                indices[i + 1] = index

        return acuracias[0], indices[0]

    def train(self, inputs, targets, epochs):
        acuracias = []
        indices = []
        for i in range(epochs):
            # Etapa Forward Propagation
            self.hidden_output = self.hidden_neuron.forward(inputs)
            self.prediction = self.output_neuron.forward(self.hidden_output)

            # Etapa Back Propagation no Neurônio de Saída
            output_error = targets - self.prediction
            self.output_neuron.back_propagation(output_error, self.learning_rate)

            # Etapa Back np Neurônio Oculto
            hidden_error = np.dot(output_error, self.output_neuron.weights.T)
            self.hidden_neuron.back_propagation(hidden_error, self.learning_rate)

            # Acurácia do Modelo
            accuracy = self.calculateAccuracy(targets, output_error)

            acuracias.append(accuracy)
            indices.append(i + 1)

            print(f"Época {i + 1} Acurácia {accuracy}")

        maior_acuracia, melhor_epoca = self.melhorEpoca(acuracias, indices)
        print(f"Melhor Época {melhor_epoca} Acurácia: {maior_acuracia}")


path = 'C:\\Users\\eduar\\OneDrive\\Documents\\Projetos Codigos\\Projetos VSCode\\Python\\LAURA LAB\\Redes_Neurais\\heart.csv'


data = pd.read_csv(path)
X = data.drop(columns=['target']).values
y = data['target'].values.reshape(-1, 1)


mlp = MLP(input_size=X.shape[1], hidden_size=5, output_size=1, learning_rate=0.1)
mlp.train(X, y, epochs=100)