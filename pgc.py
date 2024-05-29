import numpy as np
import matplotlib.pyplot as plt
import pdb

def gennet_inh_lat(input_neuron, compet_neuron):
    m11 = np.zeros((input_neuron, input_neuron))
    m12 = np.zeros((input_neuron, compet_neuron))

    m21 = 0.1 * np.ones((compet_neuron, input_neuron)) + 0.1 * np.random.rand(compet_neuron, input_neuron)

    m22 = -(0.8 * (np.ones((compet_neuron, compet_neuron)) - np.eye(compet_neuron))) + 0.6 * np.eye(compet_neuron) #Esta última linha sustitui  anterior acrescentando autapses

    w = np.block([[m11, m12], [m21, m22]])
    m22 = np.zeros_like(m22)
    w1 = np.block([[m11, m12], [m21, m22]])
    mask = w1 > np.zeros_like(w1)

    return w, mask

def imprimir_matriz_como_tabela(matriz):
    for linha in matriz:
        for elemento in linha:
            print("{:.2f}".format(elemento), end="\t")
        print()

# Gera matriz de pesos
w, mascara = gennet_inh_lat(5,3)

# Numero de neuronios (numero de linhas da matriz w)
n_neuronios = w.shape[0]

# Deslocamento da sigmoide para Inet direita
shift = 0.5 * np.ones((n_neuronios, 1))

# Fator de aprendizado
fator_aprendizado = 0.001

# Velocidade de deslocamento da sigmóide
velocidade_deslocamento = 0.025

# Número de vezes em que são apresentados todos os padrões
epocas = 1000

#Incremento no peso sináptico
incw = np.zeros_like(w)

# Saída da rede
output_antes = np.zeros((n_neuronios, 1))

# Saída da rede no instante anterior
output = output_antes.copy()

# Matriz com os padrões para o aprendizado
# Na ordem: A,B,A,C,A
P = np.array([[0.1, 0.2, 0.0, 0.0, 0.7],
              [0.0, 0.0, 0.4, 0.6, 0.8],
              [0.3, 0.1, 0.0, 0.0, 0.6],
              [0.8, 0.2, 0.0, 0.0, 0.0],
              [0.2, 0.1, 0.0, 0.0, 0.7]]).T

# Inet matriz é transposta, n_linhas = n_padroes, n_colunas = comp_padroes

# Número de padrões de entrada / Comprimento dos padrões de entrada
n_entradas, padroes = P.shape

# Numero de camadas da rede
camadas = 1

# Numero de interações que ocorreram
inter_totais = 0

imprimir_matriz_como_tabela(w)

def funcao_ativacao(ativacao, shift):
  return 1 / (1 + np.exp(-70 * (ativacao - shift)))

output_graf = np.zeros((epocas - padroes, n_neuronios))

for i in range(epocas):
    for j in range(padroes):
      output = np.zeros((n_neuronios, 1))
      output_antes = output

      PAT = P[:, j]
      output[0:n_entradas, 0] = PAT

      for k in range(camadas + 1):
        w = w + incw
        Inet = np.dot(w, output)

        output = funcao_ativacao(Inet, shift)

        output = (Inet > 0.0) * output
        output[0:n_entradas, 0] = PAT

        incw = (fator_aprendizado * (np.dot(output, output_antes.T) - ((1 + 0.05) * np.ones_like(output) * output_antes.T) * w)) * mascara
        shift = (velocidade_deslocamento * output + shift) / (1 + velocidade_deslocamento)
        output_antes = output

      if i >= (epocas - padroes):
        output_graf[inter_totais, :] = output.T
        inter_totais += 1

# pdb.set_trace()
output_graf

# r = np.arange(1, padroes + 1)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.plot_surface(r, np.arange(n_neuronios), output_graf.T, cmap='viridis')
# ax.view_init(100, 80)
# plt.show()