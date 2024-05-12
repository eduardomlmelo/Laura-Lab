import random
import pandas as df
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics

# Funções de Operações
def soma(Vf, Vi, modelo):
    soma = []
    for i in range(len(Vf))
        if modelo != 'knn' and (i == 2 or i == 3):
            soma.append(round(Vf[i] + Vi[i], 4))
        else:
            soma.append(Vf[i] + Vi[i])
    return soma

def subtrair(Vf, Vi):
    subtracao = []
    for i in range(len(Vf)):
        subtracao.append(Vf[i] - Vi[i])
    return subtracao
    
def multiplicar(v, c, modelo):
    multiplicacao = []
    for i in range(len(v)):
        if modelo != "knn" and (i == 2 or i == 3):
            multiplicacao.append(v[i] * c)
        else:
            multiplicacao.append(v[i] * c)
    return multiplicacao

# Classe de Partícula
class Particula:
    def __init__(self, velocidade, posicao):
        self.posicao = posicao
        self.velocidade = velocidade
        self.performace = None
        self.melhor_posicao = None
        self.c1 = random(0, 2)
        self.c2 = random(0, 2)
        self.melhor_performace = 0
    def prox_Vel(self, melhor_pos_geral, algoritmo):
        

# Classe do Algoritmo