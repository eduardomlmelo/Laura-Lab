from random import randint, choice

class GeneticAlgorithym:
    # Classe para o Algoritmo de busca Genético.
    # Esta Classe, deve ter três atributos: (Número de Indivíduos, Número de Populações e Chance de Mutar)
    # Além disso, deve possuir os seguinte métodos: (gerarIndividuo, gerarPopulacao, calcularFitness, selecaoTorneio, crossover, mutacao)
    num_individuos = int
    num_populacao = int
    chance_mutar = int # Percentual dado em forma inteira
    
    
    def __init__(self, num_individuos, num_populacao, chance_mutar): # Método Construtor para inicializar os valores dos Atributos da Classe
        self.num_individuos = num_individuos # Refere-se à quantidade de Indivíduos em uma População
        self.num_populacao = num_populacao # Refere-se à quantidade de Populações a serem geradas, ou seja, quantas vezes o algoritmo vai ser executado
        self.chance_mutar = chance_mutar # Refere-se à porcentagem de um 'cromossomo' ter seus valores modificados
        self.melhor_individuo = None # Atributo que vai armazenar qual o melhor Indivíduo encontrado, em todas as populações
        self.valores = [randint(1,5),randint(1,10),randint(1,15),randint(1,20),randint(1,25)] # Lista que armazena os possíveis valores para cada 'atributo' de um indivíduo

    def gerarIndividuo(self): # Método responsável por criar um Indivíduo. Em outras palavras, ele define o Indivíduo como um vetor de 5 posições e atribui valores correspondentes às suas posições

        return [randint(1,5),randint(1,10),randint(1,15),randint(1,20),randint(1,25)]
    
    def gerarPopulacao(self, num_individuos): #Método responsável por cria uma População, ou seja, um conjunto de Indivíduos

        populacao = []
        for i in range(num_individuos):
            populacao.append(self.gerarIndividuo)
        return populacao
    
    def calcularFitness(self, individuo): # Método que calcula o Fitness de um indivíduo, através da função de Soma
        
        return sum(individuo)/75
    
    def torneio(self, populacao): # Método que determina dois melhores indivíduos, um Pai e uma Mãe, escolhidos aleatoriamente da população
        selecaoH = [choice(populacao), choice(populacao)] # Cria uma lista de dois elementos, os quais, por suas vezes, são elementos aleatórios do conjunto 'populacao'
        selecaoM = [choice(populacao), choice(populacao)]

        pai = [] # Lista vazia que guardará o elemento de maior Fitness do vetor 'selecaoH'
        if(self.calcularFitness(selecaoH[0]) >= self.calcularFitness(selecaoH[1])): # Verifica qual elemento possui o maior Fitness
            pai.append(selecaoH[0]) # Adiciona ao vetor 'pai' o elemento em questão
        else:
            pai.append(selecaoH[1])

        mae = [] # Lista vazia que guardará o elemento de maior Fitness do vetor 'selecaoM'
        if(self.calcularFitness(selecaoM[0]) >= self.calcularFitness(selecaoM[1])):
            mae.append(selecaoM[0])
        else:
            mae.append(selecaoM[1])

        return pai, mae
    
    def crossover(self, pai, mae): # Método que gera dois filhos por casal, utilizando uma 'mescla genética' dos pais
        posicao = randint(1,4)
        filho = pai[:posicao] + mae[posicao:] # Filho recebe a primeira parte do pai e a, segunda 
        filha = mae[:posicao] + pai[posicao:]

        return filho, filha

    def mutacao(self, chance_mutar, filho, filha, familia_m): #Método responsável por redefinir os valores dos 'atributos' de um indivíduo, com base na sua 'chance de mutação'
        chance = 0

        for i in range(5):
            chance = randint(1,100)
            if(chance <= chance_mutar):
                filho[i] = randint(1,self.valores[i])
                filha[i] = randint(1, self.valores[i])
                familia_m.append(filho)
                familia_m.append(filha)

        return familia_m