import torch

import n3ml.network
import n3ml.connection


class Ponulak:
    def __init__(self,
                 synapse: n3ml.connection.Synapse,
                 lr: float = 0.01,
                 a_d: float = 0.0,
                 a_l: float = 0.0,
                 A_d: float = 1.0,
                 A_l: float = 1.0,
                 tau_d: float = 1.0,
                 tau_l: float = 1.0) -> None:
        self.synapse = synapse
        self.lr = lr
        self.a_d = a_d
        self.a_l = a_l
        self.A_d = A_d
        self.A_l = A_l
        self.tau_d = tau_d
        self.tau_l = tau_l

    def step(self,
             t: int,
             input_spike_train: torch.Tensor,
             learning_spike: torch.Tensor,
             desired_spike: torch.Tensor) -> None:
        # Reverse the order at dimension 0
        input_spike_train = torch.flip(input_spike_train, dims=[0])
        # print(input_spike_train.size(0))
        del_W = torch.zeros_like(self.synapse.w)
        for k in range(input_spike_train.size(1)):  # number of input neurons
            for i in range(learning_spike.size(0)):  # number of learning neurons
                del_w = 0
                if learning_spike[i] > 0.5:
                    del_w_l = self.a_l
                    for j in range(input_spike_train.size(0)):
                        s_l = torch.tensor(t - j)
                        # s_l = torch.tensor(j)
                        # print("depression: {}".format(Ponulak.depress(s_l, self.A_l, self.tau_l)))
                        # print("s_l: {} - A_l: {} - tau_l: {} - del_w_l: {}".format(s_l, self.A_l, self.tau_l, Ponulak.depress(s_l, self.A_l, self.tau_l)))
                        del_w_l += Ponulak.depress(s_l, self.A_l, self.tau_l) * input_spike_train[j][k]
                    # print("del_w_l: {}".format(del_w_l))
                    del_w += del_w_l
                if desired_spike[i] > 0.5:
                    del_w_d = self.a_d
                    for j in range(input_spike_train.size(0)):
                        s_d = torch.tensor(t - j)
                        # s_d = torch.tensor(j)
                        # print("potentiation: {}".format(Ponulak.potentiate(s_d, self.A_d, self.tau_d)))
                        del_w_d += Ponulak.potentiate(s_d, self.A_d, self.tau_d) * input_spike_train[j][k]
                    # print("del_w_d: {}".format(del_w_d))
                    del_w += del_w_d
                # print("del_w: {}".format(del_w))
                del_W[i][k] = del_w
        # print(del_W.numpy() * self.lr)
        self.synapse.w += del_W * self.lr

    @classmethod
    def potentiate(cls, s_d, A_d, tau_d):
        if s_d > 0:
            return A_d * torch.exp(-s_d/tau_d)
        return 0

    @classmethod
    def depress(cls, s_l, A_l, tau_l):
        if s_l > 0:
            return -A_l * torch.exp(-s_l/tau_l)
        return 0


def y(t_j, t_i, d_k, tau):
    if t_j >= 0 and t_i >= 0:
        t = t_j - t_i - d_k
        if t >= 0:
            return t * torch.exp(1 - t / tau) / tau
    return 0


def dydt(t_j, t_i, d_k, tau):
    # w.r.t t_j
    if t_j >= 0 and t_i >= 0:
        t = t_j - t_i - d_k
        if t >= 0:
            return torch.exp(1 - t / tau) / tau - t * torch.exp(1 - t / tau) / (tau ** 2)
    return 0


def dydt2(t_j, t_i, d_k, tau):
    # w.r.t t_i
    if t_j >= 0 and t_i >= 0:
        t = t_j - t_i - d_k
        if t >= 0:
            return -torch.exp(1 - t / tau) / tau + t * torch.exp(1 - t / tau) / (tau ** 2)
    return 0


class Bohte:
    def __init__(self):
        pass

    def step(self, model, spiked_input, spiked_label, epoch):
        # print(model)

        # lr = 0.0001  #
        # lr = 0.01
        lr = 0.0075

        layer = []
        spike_time = [spiked_input]  # with input spike

        for l in model.layer.values():
            layer.append(l)
            spike_time.append(l.s)
        layer.reverse()
        spike_time.reverse()

        error = []

        for l in range(len(layer)):
            if l == 0:
                # print("last layer")
                dldx = torch.zeros(layer[l].out_neurons)
                for j in range(layer[l].out_neurons):
                    numer = (layer[l].s[j]-spiked_label[j])
                    denom = 0
                    for i in range(layer[l].in_neurons):
                        for k in range(layer[l].delays):
                            denom += (layer[l].w[j, i, k]*dydt(spike_time[l][j], spike_time[l+1][i], layer[l].d[k], layer[l].tau_rc))
                    dldx[j] = -numer/(denom+1e-15)
                error.append(dldx)
                dxdw = torch.zeros_like(layer[l].w)
                for j in range(layer[l].out_neurons):
                    for i in range(layer[l].in_neurons):
                        for k in range(layer[l].delays):
                            dxdw[j, i, k] = y(spike_time[l][j], spike_time[l+1][i], layer[l].d[k], layer[l].tau_rc)
                g = torch.zeros_like(layer[l].w)
                for j in range(layer[l].out_neurons):
                    for i in range(layer[l].in_neurons):
                        for k in range(layer[l].delays):
                            g[j, i, k] = -lr * dxdw[j, i, k] * dldx[j]
                # print(g.detach().numpy())
                layer[l].w += g
                # print(layer[l].w.detach().numpy())
                layer[l].w[:] = torch.clamp(layer[l].w, 0.02, 1)  #
            else:
                # print("intermediate layer")
                dldx = torch.zeros(layer[l].out_neurons)
                for i in range(layer[l].out_neurons):
                    numer = 0
                    for j in range(layer[l-1].out_neurons):
                        sum = 0
                        for k in range(layer[l-1].delays):
                            sum += layer[l-1].w[j, i, k] * dydt2(spike_time[l-1][j], spike_time[l][i], layer[l-1].d[k], layer[l-1].tau_rc)
                        numer += error[-1][j] * sum
                    denom = 0
                    for h in range(layer[l].in_neurons):
                        for m in range(layer[l].delays):
                            denom += layer[l].w[i, h, m] * dydt(spike_time[l][i], spike_time[l+1][h], layer[l].d[m], layer[l].tau_rc)
                    dldx[i] = -numer/(denom+1e-15)
                error.append(dldx)
                dxdw = torch.zeros_like(layer[l].w)
                for i in range(layer[l].out_neurons):
                    for h in range(layer[l].in_neurons):
                        for m in range(layer[l].delays):
                            dxdw[i, h, m] = y(spike_time[l][i], spike_time[l+1][h], layer[l].d[m], layer[l].tau_rc)
                g = torch.zeros_like(layer[l].w)
                for i in range(layer[l].out_neurons):
                    for h in range(layer[l].in_neurons):
                        for m in range(layer[l].delays):
                            g[i, h, m] = -lr * dxdw[i, h, m] * dldx[i]
                # print(g)
                layer[l].w += g
                layer[l].w[:] = torch.clamp(layer[l].w, 0.02, 1)


class TavanaeiAndMaida:
    def __init__(self,
                 model: n3ml.network.Network,
                 lr: float = 0.0005) -> None:
        self.model = model
        self.lr = lr

    def step(self, spike_buffer, spiked_label, label):
        """"""
        """
            텐서로 변환된 spike_buffer[b]의 크기는 [epsilon, # neurons]와 같다.
        """
        buffer = {}
        for b in spike_buffer:
            buffer[b] = torch.stack(spike_buffer[b])

        if spiked_label[label] > 0.5:  # target neuron fires at that time
            in_grad = torch.zeros(self.model.fc2.out_neurons)
            for i in range(self.model.fc2.out_neurons):
                if i == label:
                    if torch.sum(buffer['fc2'][:, i]) < 1:
                        in_grad[i] = 1
                else:
                    if torch.sum(buffer['fc2'][:, i]) > 0:
                        in_grad[i] = -1

            # print(in_grad.numpy())

            # Compute propagated error in line 17-18
            e = torch.matmul(in_grad, self.model.fc2.w) * (torch.sum(buffer['fc1'], dim=0) > 0)

            # Update the weights in last layer in line 19
            updates = torch.ger(in_grad, torch.sum(buffer['fc1'], dim=0))
            # print(updates)
            self.model.fc2.w += updates * self.lr

            updates = torch.ger(e, torch.sum(buffer['inp'], dim=0))
            self.model.fc1.w += updates * self.lr
            # print(updates)
