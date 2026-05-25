import math
import torch


class SGD:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    @torch.no_grad()
    def step(self):
        for param in self.params:
            if param.grad is not None:
                param -= self.lr * param.grad


class Momentum:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [torch.zeros_like(param) for param in self.params]

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    @torch.no_grad()
    def step(self):
        for param, velocity in zip(self.params, self.velocity):
            if param.grad is not None:
                velocity.mul_(self.momentum).add_(param.grad)
                param -= self.lr * velocity


class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [torch.zeros_like(param) for param in self.params]
        self.v = [torch.zeros_like(param) for param in self.params]

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    @torch.no_grad()
    def step(self):
        self.t += 1
        for param, m, v in zip(self.params, self.m, self.v):
            if param.grad is None:
                continue

            grad = param.grad
            m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)

            m_hat = m / (1 - math.pow(self.beta1, self.t))
            v_hat = v / (1 - math.pow(self.beta2, self.t))
            param -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
