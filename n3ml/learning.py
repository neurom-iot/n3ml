import torch


class LearningRule:
    def __init__(self):
        pass


class PostPre(LearningRule):
    def __init__(self,
                 connection,
                 lr=(1e-10, 1e-3)):
        super().__init__()

        self.connection = connection
        # lr은 tuple 타입으로 크기가 2가 된다. lr[0]는 presynaptic spike에 대한 weight change에서 사용되고
        # lr[1]은 postsynaptic spike에 대한 weight change에서 사용된다.
        self.lr = lr

    def run(self) -> None:
        # prev = self.connection.w.clone()

        # Compute weight changes for presynaptic spikes
        s_pre = self.connection.source.s.unsqueeze(1)
        x_post = self.connection.target.x.unsqueeze(0)
        self.connection.w -= self.lr[0] * torch.transpose(torch.mm(s_pre, x_post), 0, 1)

        # Compute weight changes for postsynaptic spikes
        s_post = self.connection.target.s.unsqueeze(1)
        x_pre = self.connection.source.x.unsqueeze(0)
        self.connection.w += self.lr[1] * torch.mm(s_post, x_pre)

        # Clamp synaptic weight
        self.connection.w[:] = torch.clamp(self.connection.w, min=self.connection.w_min, max=self.connection.w_max)

        # print("s_pre.sum: {} - s_post.sum: {}".format(s_pre.sum(), s_post.sum()))

        # print("s_pre:\n{}".format(s_pre))
        # print("x_pre:\n{}".format(x_pre))
        # print("s_post:\n{}".format(s_post))
        # print("x_post:\n{}".format(x_post))

        # now = self.connection.w.clone()
        # print("diff of prev and now: {}".format((now - prev).sum()))


class BaseLearning:
    def __init__(self):
        pass


class ReSuMe(BaseLearning):
    def __init__(self):
        super().__init__()
