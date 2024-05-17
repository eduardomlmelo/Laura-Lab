import pandas as df
import numpy as np
from random import randint


class Neuronio:
    def __init__(self, min_weight, max_weight, learning_value, datapath):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.learning_rate = learning_value
        self.dataset_path = datapath

        self.bias = randint(0, 2)
        self.weights = []
    def randomizeWeights(self, lenght):
        self.weights = np.random.randn(lenght, 1)

    def storageWeights(self):
        return self.weights
    def sigmoid(self, x):
        # Função de ativação sigmoid
        return 1 / 1 + np.exp(-x)
    def sigmoid_derivative(self, x):
        # Derivada da função sigmoid
        return x * (1 - x)
    def calcularMatrizPonderada(self, features):
        return np.dot(features, self.weights) + self.bias
    def forward(self, features):  # Operação responsável por receber dados, processá-los e retorná-los como saída

        # Transformação de dados
        matriz_ponderada = self.calcularMatrizPonderada(features)
        matriz_sigmoid = self.sigmoid(matriz_ponderada)

        return matriz_sigmoid

    def updateWeights(self, accuracy):
        self.weights += self.learning_rate * self.sigmoid_derivative(accuracy)


class Cerebro(Neuronio):
    def __init__(self, min_weight, max_weight, learning_value, datapath, input_size, hidden_size, output_size):
        super().__init__(min_weight, max_weight, learning_value, datapath)
        self.dataset_path = datapath
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.min_error = 0.5

        self.executar()
    def loadData(self):

        # Carregar o DataFrame a partir do arquivo CSV
        data_base = df.read_csv(self.dataset_path)

        # Extraindo features e target
        X = data_base.drop(columns=['target'])  # Todas as colunas exceto 'target'
        y = (data_base['target'])
        print((len(X.values)))
        return X.values, y.values, len(np.transpose(X.values))

    def calcularAcuracia(self, targets, inputs):
        return sum(inputs) / sum(targets + inputs)
    def train(self, features, targets, lenght, input_size, hidden_size, output_size):
        oculto = Neuronio(self.min_weight, self.max_weight, self.learning_rate, self.dataset_path)
        oculto.randomizeWeights(lenght)

        exit_neuron = Neuronio(self.min_weight, self.max_weight, self.learning_rate, self.dataset_path)
        '''
        exit_neuron.randomizeWeights(lenght)
        exit_neuron.weighs = np.transpose(exit_neuron.weights)
        '''
        contador = 0
        accuracy = 0
        for i in range(input_size):

            for j in range(hidden_size):

                for k in range(output_size):

                    hidden = oculto.forward(features) # 1025 / 1
                    hidden = oculto.sigmoid(hidden)
                    saida = exit_neuron.sigmoid(hidden) # 1025 / 1

                    new_output = []
                    for i in range(len(saida)):
                        new_output.append(np.sum(saida[i]))
                    print(new_output)
                    accuracy = self.calcularAcuracia(targets, new_output)

                    oculto.updateWeights(accuracy)

                    contador += 1

        print(f"Acurácia: \n {accuracy} \n")

    def executar(self):
        X, y, size = self.loadData()
        self.train(X, y, size, self.input_size, self.hidden_size, self.output_size)


path = 'C:\\Users\\eduar\\OneDrive\\Documents\\Projetos Codigos\\Projetos VSCode\\Python\\LAURA LAB\\Redes_Neurais\\heart.csv'

min_weight = -1
max_weight = 1
learning_value = 0.4
num_entradas = 3
num_ocultas = 5
num_saidas = 3

modelo = Cerebro(min_weight, max_weight, learning_value, path, num_entradas, num_ocultas, num_saidas)