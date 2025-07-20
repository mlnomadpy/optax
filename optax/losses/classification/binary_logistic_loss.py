from .sigmoid_binary_cross_entropy import sigmoid_binary_cross_entropy

def binary_logistic_loss(logits, labels):
    return sigmoid_binary_cross_entropy(logits, labels)
