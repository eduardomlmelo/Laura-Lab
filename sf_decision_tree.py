import pandas as pd
# from sklearn.datasets import load_data_base
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

from ucimlrepo import fetch_ucirepo
# Incia uma contagem de tempo no começo do processo
t1 = time.perf_counter()
# Armazenar toda a Base de Dados que vamos trabalhar
data_base = fetch_ucirepo(id=53) # Base de dados data_base
# Guardar em X, as colunas de Features da Base de Dados
X = data_base.data.features
# Guardar em y, a coluna de  Targets da base de Dados
y = data_base.data.targets
# Cria as variáveis de Treino e Teste, definindo o tamanho do Espaço de Treino e o fator aleatório de separação
x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=1)
# Define qual modelo vai utilizar, no caso, cria o modelo de Árvore de Decisão
tree_classifier_model = DecisionTreeClassifier(random_state=1)
# Treina o modelo com a função '.fit'
tree_classifier_model.fit(x_treino, y_treino)
# Faz uma previsão com o modelo
y_previsto = tree_classifier_model.predict(x_teste)
# Calcula a Accuracy do modelo
accuracy = accuracy_score(y_teste, y_previsto)
# Marcador de tempo do Fim do processo
t2 = time.perf_counter()
# Calcula o tempo total de execução do modelo
tempo = t2-t1
# Mostra o resultado do modelo, a Acurácia + Tempo
print(f"Acurácia do Modelo Árvore de Decisão: {accuracy:.2f}")
print(f"Tempo de execução: {tempo:.2f}")
print(f"Parâmetros utilizados: id de dados({53}), espaço de teste({0.3}), fator de aleatoriedade({1})")