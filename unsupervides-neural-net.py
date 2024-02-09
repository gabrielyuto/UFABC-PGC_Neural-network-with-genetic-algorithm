import numpy as np
import matplotlib.pyplot as plt
import pdb;

def gennet_inh_lat(input_neuron, compet_neuron):
    m11 = np.zeros((input_neuron, input_neuron))
    m12 = np.zeros((input_neuron, compet_neuron))

    m21 = 0.1 * np.ones((compet_neuron, input_neuron)) + 0.1 * np.random.rand(compet_neuron, input_neuron)
    m22 = -(0.8 * (np.ones((compet_neuron, compet_neuron)) - np.eye(compet_neuron))) + 0.6 * np.eye(compet_neuron)
    breakpoint()
    w = np.block([[m11, m12], [m21, m22]])
    m22 = np.zeros_like(m22)
    w1 = np.block([[m11, m12], [m21, m22]])
    mask = w1 > np.zeros_like(w1)

    return w, mask

w, mascara = gennet_inh_lat(5,3)


w, mascara = gennet_inh_lat(5,3) #Gera matriz de pesos
n_neuronios = w.shape[0] #Numero de neuronios (numero de linhas da matriz w)

shift = 0.5 * np.ones((n_neuronios, 1)) #Deslocamento da sigmoide para Inet direita
fator_aprendiz = 0.001 #Fator de aprendizado
velocidade_deslocamento = 0.025 # Velocidade de deslocamento da sigmóide
epocas = 10000 #Número de vezes em que são apresentados todos os padrões

# Inicialização das variáveis
incw = np.zeros_like(w) #Incremento no peso sináptico
output_antes = np.zeros((n_neuronios, 1)) #Saída da rede
output = output_antes.copy() # Saída da rede no instante anterior

ipdb.set_trace()


# Matriz com os padrões para o aprendizado
# Na ordem: A,B,A,C,A
P = np.array([[0.1, 0.2, 0.0, 0.0, 0.7],
              [0.0, 0.0, 0.4, 0.6, 0.8],
              [0.3, 0.1, 0.0, 0.0, 0.6],
              [0.8, 0.2, 0.0, 0.0, 0.0],
              [0.2, 0.1, 0.0, 0.0, 0.7]]).T

# Inet matriz é transposta, n_linhas = n_padroes, n_colunas = comp_padroes

n_entradas, padroes = P.shape #Número de padrões de entrada / Comprimento dos padrões de entrada
camadas = 2 #Numero de camadas da rede
inter_totais = 1 #Numero de interações que ocorreram

# output_graf = np.zeros((epocas - padroes, padroes))

for i in range(epocas): #Para cada iteração
    for j in range(padroes): # Para cada padrão de entrada
        output = np.zeros((n_neuronios, 1))
        output_antes = output.copy()

        PAT = P[:, j]
        output[:n_entradas, 0] = PAT

        for k in range(camadas + 1):
            w += incw
            Inet = np.dot(w, output)
            output = 1 / (1 + np.exp(-70 * (Inet - shift)))
            output = (Inet > 0.0) * output
            output[:n_entradas, 0] = PAT
            incw = (fator_aprendiz * (np.dot(output, output_antes.T) - ((1 + 0.05) * np.ones_like(output) * output_antes.T) * w)) * mascara
            shift = (velocidade_deslocamento * output + shift) / (1 + velocidade_deslocamento)
            output_antes = output

        if i >= (epocas - padroes):
            output_graf[inter_totais - padroes, :] = output.squeeze()
            inter_totais += 1

# r = np.arange(1, padroes + 1)
# plt.figure()
# plt.plot(r, output_graf.T)
# plt.show()

