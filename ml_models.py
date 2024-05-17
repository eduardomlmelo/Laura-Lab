import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import fetch_openml
from ucimlrepo import fetch_ucirepo

data_base = fetch_ucirepo(id=45)
# data_base = fetch_openml(name='iris', version=1)
# data_base = pd.read_csv("diabetes.csv")

# Variáveis da ucirepo
X = data_base.data.features
y = data_base.data.targets

'''
# Variáveis da openMl
X = data_base.data
y = data_base.target
'''
class DecisionTree:
    def __init__(self, features, targets, test_size, random_state):
        self.features = features
        self.targets = targets
        self.test_size = test_size
        self.random_state = random_state

        self.executar()

    def executar(self):
        x_treino, x_teste, y_treino, y_teste = self.TrainingVariables(self.features, self.targets, self.test_size, self.random_state)
        tree = self.ModelFit(x_treino, y_treino, self.random_state)
        y_previsto = self.ModelPredict(tree, x_teste)
        acuracia = self.TreeAccuracy(tree, y_teste, y_previsto)
    

    def TrainingVariables(self, features, targets, test_size, random_state):
        x_train = [], x_test = [], y_train = [], y_test = []
        x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size, random_state)
        return x_train, x_test, y_train, y_test
    

    def ModelFit(self, x_treino, y_treino, random_state):
        tree_classifier = DecisionTreeClassifier(random_state= random_state)
        tree_classifier.fit(x_treino, y_treino)
        return tree_classifier

    
    def ModelPredict(self, tree_model, x_teste):
        y_predicted = tree_model.predict(x_teste)
        return y_predicted
    

    def TreeAccuracy(self, tree_model, y_predicted, y_try):
        accuracy = tree_model.accuracy_score(y_try, y_predicted)
        return accuracy
    
    
modelo_arvore = DecisionTree(X, y, 0.3, 42)
print(modelo_arvore.TreeAccuracy)
