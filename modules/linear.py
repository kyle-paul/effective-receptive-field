import torch
import torch.nn as nn
import numpy as np
from utils import CrossEntropy

class Linear:
    def __init__(self, in_features, out_features, bias=True):
        limit = np.sqrt(6 / in_features)
        self.weights = np.random.uniform(-limit, limit, (in_features, out_features))
    
    def forward(self, x):
        self.x = x
        return np.dot(x, self.weights)
    
    def backward(self, grad_logits):
        self.x_T = self.x.reshape(x.shape[1], x.shape[0])
        self.grad_weights = grad_logits * self.x_T
        self.grad_x = np.dot(grad_logits, self.weights.T)
        return self.grad_x, self.grad_weights
        

if __name__ == "__main__":

    x = np.array([1, 2]).reshape(1, -1)
    targets = np.array([2]) 

    fc = Linear(in_features=2, out_features=3)
    logits = fc.forward(x)
    print(logits)

    ce = CrossEntropy(num_classes=3)
    loss = ce.forward(logits, targets)
    print(loss)

    grad_logits = ce.backward()
    grad_x, grad_weights = fc.backward(grad_logits)
    print(grad_logits)
    print(grad_weights)
    print(grad_x)
    print()

    x_ = torch.tensor([1, 2]).unsqueeze(0).to(torch.float32)
    x_ = x_.requires_grad_(True)
    targets = torch.tensor([2])
    fc_ = nn.Linear(in_features=2, out_features=3, bias=False)
    fc_.weight = torch.nn.Parameter(torch.tensor(fc.weights.T).to(torch.float32))
    logits_ = fc_(x_)
    logits_.retain_grad()
    print(logits_)
    
    ce_ = nn.CrossEntropyLoss()
    loss_ = ce_(logits_, targets)
    print(loss_)

    loss_.backward()
    print(logits_.grad)
    print(fc_.weight.grad)
    print(x_.grad)