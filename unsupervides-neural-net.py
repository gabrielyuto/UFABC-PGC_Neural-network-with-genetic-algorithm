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
