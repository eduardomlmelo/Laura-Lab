import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from ucimlrepo import fetch_ucirepo

# Inicia um marcador de tempo no início do processo
t1 = time.perf_counter()
# Guarda os dados da Base de Dados utilizada
data_base = fetch_ucirepo(id=53)
# Guarda as colunas das Features da base de dados
X = data_base.data.features
# Guarda a coluna dos Targets da base de dados
y = data_base.data.targets
# Cria as variáveis de Treino e Teste, define o Espaço de Treino e Fator Aleatório, com a função 'train_test_split'
x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=1)
# Variável de Padronização do dados armazenados
scaler = StandardScaler()
# Padronização dos dados de Treino, com a 'scaler', usando a função '.fit_transform'
x_treino = scaler.fit_transform(x_treino)
# Padronização dos dados de Teste, com a 'scaler', usando a função '.fit_transform'
x_teste = scaler.fit_transform(x_teste)
# Define a variável de Vizinhos Considerados
k = 5
# Define qual modelo será utilizado, no caso, o KNN
knn_classifier = KNeighborsClassifier(n_neighbors=k)
# Treina o modelo, com a função '.fit'
knn_classifier.fit(x_treino, y_treino)
# Faz a previsão do modelo
y_previsto = knn_classifier.predict(x_teste)
# Calcula a Accuracy do modelo
accuracy = accuracy_score(y_teste, y_previsto)
# Marcador de tempo do final do processo
t2 = time.perf_counter()
# Calcula o tempo de execução do modelo
tempo = t2-t1

print(f"Acurácia do Modelo KNN: {accuracy:.2f}")
print(f"Tempo de execução do Modelo: {tempo:.2f}")
print(f"Parâmetros utilizados: vizinhos considerado({k}), espaço de teste({0.2}), fator aleatório({1})")

print(" ")
print("        Especificações da Base de Dados Utilizada")
print(data_base.metadata)
print(" ")
print("        Variáveis da Base de Dados")
print(data_base.variables)