########################################################## README ###########################################################
# STDP curve and weight update rule
##############################################################################################################################


import numpy as np
from matplotlib import pyplot as plt
from PhDSpikingNet.Implementation.Training.parameter import parameter


# STDP learning curve
def stdpLearningWindow(delta_t):
    """
    This function describes the STDP Learning Curve

    :param delta_t: Time different between Pre-neuron firing time and Post-neuron firing time delta_t = (t_post - t_pre)

    :return: The weight change - delta_w
    """
    if delta_t > 0:
        delta_w = parameter.A_plus * np.exp(-float(delta_t) / parameter.tau_plus)
    if delta_t < 0:
        delta_w = parameter.A_minus * np.exp(float(delta_t) / parameter.tau_minus)

    return delta_w


# STDP weight Update
def weightUpdate(w, delta_w):
    if delta_w < 0:  # Depression
        new_w= w + (w*delta_w)
    elif delta_w > 0:  # Potentiation
        new_w= w + ((parameter.wmax-w)*delta_w)
    return new_w


# STDP UpdateRule
def stdpUpdate(delta_t, currentWeight):
    delta_w = stdpLearningWindow(delta_t)
    newWeight = weightUpdate(currentWeight, delta_w)
    return newWeight




def stdpSimulation():
    '''
    This function is used to plot the STDP Curve with a given range of delta_t
    :return: STDP Learning Window Figure
    '''
    if parameter.tau_plus >= parameter.tau_minus:
        deltaRange = parameter.tau_plus
    else:
        deltaRange = parameter.tau_minus
    positive_deltat = np.linspace (0,deltaRange*3)
    positivedelta_w = parameter.A_plus*(np.exp(-positive_deltat/ parameter.tau_plus))
    negative_deltat2 = np.linspace(-(deltaRange*3), 0)
    negativedelta_w = parameter.A_minus * (np.exp(negative_deltat2 / parameter.tau_minus))
    plt.plot(positive_deltat, positivedelta_w, label='Weight Potentiation')
    plt.plot(negative_deltat2, negativedelta_w, label='Weight Depression')
    plt.axhline(y=0, color='black')
    plt.axvline(x=0, color='black')
    plt.xlabel('Delta_t in milisecond')
    plt.ylabel('Delta_W')
    plt.title('STDP Learning Window')
    plt.legend()
    plt.show()


if __name__ == '__main__':

    stdpSimulation()

    # delta_w = stdpLearningWindow(-10)
    # print(delta_w)
    # w = weightUpdate(0.5, delta_w)
    # print(w)


