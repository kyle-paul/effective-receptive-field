import numpy as np

class CrossEntropy:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.grad = None
    
    def one_hot_encode(self, targets):
        one_hot = np.zeros((targets.shape[0], self.num_classes))
        one_hot[np.arange(targets.shape[0]), targets] = 1
        return np.float32(one_hot)
        
    def logsumexp(self, x):
        c = x.max(axis=1, keepdims=True)
        return c + np.log(np.sum(np.exp(x - c), axis=1, keepdims=True))

    def softmax(self, x):
        return np.exp(x - self.logsumexp(x))

    def forward(self, logits, targets):
        self.logits = logits
        self.probs = self.softmax(logits)
        self.encoded_targets = self.one_hot_encode(targets)
        result = self.encoded_targets * np.log(self.probs) + (1 - self.encoded_targets) * np.log(1 - self.probs)
        loss = -np.sum(result)
        return loss / 2
    
    def backward(self):
        self.grad = self.probs - self.encoded_targets
        return self.grad