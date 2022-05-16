import torch

def generate_class_mask(pred, classes):
    '''
    generate class mask
    :param pred: predicted mask（H*W）
    :param classes: classes [0,1,2,...]
    :return: mask（H*W）
    '''
    pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
    N = pred.eq(classes).sum(0)
    return N

