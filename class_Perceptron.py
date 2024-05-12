import pandas as df
import numpy as np
from random import randint

class Perceptron:

    def __init__(self, min_weight, max_weight, threshold, learning_rate, dataset_path):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.dataset_path = dataset_path
        self.weights = []


    def loadData(self):
        # Carregar o DataFrame a partir do arquivo CSV
        data_base = df.read_csv(self.dataset_path)
        # Extraindo features e target
        X = data_base.drop(columns=['target'])  # Todas as colunas exceto 'target'
        y = data_base['target']  # Apenas a coluna 'target'
        # Inicializar os pesos com valores aleatórios
        self.weights = [randint(self.min_weight, self.max_weight) for _ in range(len(X.columns))]
        return X.values, y.values


    def calcularPonderada(self, features):
        # Calcular a soma ponderada dos features e pesos
        return np.dot(features, self.weights)


    def stepActivation(self, ponderada):
        # Função de ativação (step function)
        return 1 if ponderada >= self.threshold else 0
    

    def sigActivation(self, ponderada, feature):
        # Função de ativação (sigmoid)
        return 1 / (1 + np.exp(-feature)) if ponderada <= self.threshold else 0
    

    def updateWeights(self, features, target, predicted):
        # Atualizar os pesos com base no erro
        error = target - predicted
        self.weights += self.learning_rate * error * features


    def singleTrain(self, X, y):
        total_error = 0
        for feature, target in zip(X, y):
            ponderada = self.calcularPonderada(feature)
            predicted = self.stepActivation(ponderada)
            self.updateWeights(feature, target, predicted)
            total_error += abs(target - predicted)
        print(f"Single total error: {total_error}")

    
    def multiTrain(self, X, y):
        for i in range(100):
            total_error = 0
            for feature, target in zip(X, y):
                ponderada = self.calcularPonderada(feature)
                predicted = self.stepActivation(ponderada)
                self.updateWeights(feature, target, predicted)
                total_error += abs(target - predicted)
            print(f"Época {i + 1}: Total Error = {total_error}")


    def executar(self):
        X, y = self.loadData()
        self.singleTrain(X, y)
        #self.multiTrain(X, y)

# Caminho para o arquivo CSV
dataset_path = "C:\\Users\\eduar\\OneDrive\\Documents\\Projetos Codigos\\Projetos VSCode\\Python\\LAURA LAB\\Modelos de Machine Learning\\heart.csv"

# Parâmetros do Perceptron
min_weight = -1
max_weight = 1
threshold = 0
learning_value = 0.1

# Criar e executar o Perceptron
perceptron = Perceptron(min_weight, max_weight, threshold, learning_value, dataset_path)
perceptron.executar()
