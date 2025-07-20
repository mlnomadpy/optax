from .sparsemax_loss import sparsemax_loss

def binary_sparsemax_loss(logits, labels):
    return sparsemax_loss(logits, labels)
