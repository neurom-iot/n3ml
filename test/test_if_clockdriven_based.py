
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

def draw_spikes(spikes):
    print('---------------------')
    for i in range (spikes.shape[1]):
        info = spikes[:,i]

        print(info)

        for j in range (len(info)):
            if info[j] ==1:
                plt.eventplot([i], lineoffsets=j)
    plt.xticks(np.arange(0, spikes.shape[1]))
    plt.xlabel('time steps')
    plt.yticks(np.arange(0, spikes.shape[0]))
    plt.ylabel('index')
    plt.show()

def draw_potential(poten, time):
    figure, axis = plt.subplots(poten.shape[0], 1)

    for i in range ((poten.shape[0]-1),-1,-1):
        print(i)
        axis[i].plot(time, poten[i])

        # axis[i].set_title("potential_" + str(i)
    plt.setp(axis, xticks= np.arange(0,20,1) )

    plt.show()

    pass

def clock_driven_based_app(opt):
    print(opt)
    from n3ml.population import IF1d

    #generate time_line
    s_time = np.arange(0,opt.sim_time,1)


    #Initilize IF1d population
    if1d_pop = IF1d(neurons=opt.n_neurons, v_th= opt.v_th, tau_ref=opt.ref)

    weights = torch.tensor([[0.9077, 0.0767, 0.4668],
                            [0.5156, 0.5879, 0.9246],
                            [0.2729, 0.6814, 0.9235]])

    #generate input spikes
    input = torch.zeros(opt.n_neurons,opt.sim_time)
    for i in range (opt.n_neurons):
        for j in range(opt.sim_time):

            if j>0 and j%2 ==0:
                input[0][j] = 1
            elif j>0 and j%3 ==0:
                input[1][j] = 1
            elif j>0 and j%5 ==0:
                input[2][j] = 1

    draw_spikes(input)

    op_spikes = torch.zeros(opt.n_neurons,opt.sim_time)
    op_v = torch.zeros(opt.n_neurons,opt.sim_time)
    for t in s_time:
        #generate current
        x = torch.matmul(input[:,t], weights)

        o_spike = if1d_pop.run(x)
        op_spikes[:, t] = o_spike
        op_v[:,t] = if1d_pop.v

    draw_potential(op_v, s_time)
    draw_spikes(torch.flip(op_spikes,[0,]))

    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_inputs', default=3, type=int)
    parser.add_argument('--n_neurons', default=3, type=int)
    parser.add_argument('--sim_time', default=20, type=int)
    parser.add_argument('--ref', default=2, type=int)
    parser.add_argument('--resting', default=0., type=float)
    parser.add_argument('--v_th', default=3., type=float)
    parser.add_argument('--dt', default=1, type=int)

    clock_driven_based_app(parser.parse_args())






