import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Marcador de tempo Inicial
t1 = time.perf_counter()
# Guardar toda a Base de Dados
data_base = fetch_openml(name='data_base', version=1)
# Guardar as colunas das Features
X = data_base.data
# Guardar a coluna Target
y = data_base.target
# Criar as variáveis de Treino e Teste; Definir o Espaço de Treino e Fator Aleatório
x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=1)
# Variável de Padronização dos dados
scaler = StandardScaler()
# Padronização dos dados de Treino
x_treino = scaler.fit_transform(x_treino)
# Padronização dos dados de Teste
x_teste = scaler.fit_transform(x_teste)
# Definição da Variável 'k', dicionário para 'kernel'
k = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
# Definição do Indivíduo
x = [1, 0 ,2]
# Criação do modelo utilizado, no caso, o SVM
SVM_classifier = SVC(C=x[0], kernel=k[x[1]], degree=x[2])
# Treinamento do modelo
SVM_classifier.fit(x_treino, y_treino)
# Previsão do modelo
y_previsto = SVM_classifier.predict(x_teste)
# Calcular a Acurácia do modelo
accuracy = accuracy_score(y_teste, y_previsto)
# Marcador de tempo Final
t2 = time.perf_counter()
# Calcula o tempo de execução
tempo = t2-t1

print("Acurácia: ", accuracy)
print("Tempo de execução: ", tempo)