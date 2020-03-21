import nets
def get_net(bn_type):
    if bn_type =="no":
        return nets.NeuralNet
    elif bn_type=="torch bn":
        return nets.NeuralNetTorchNorm
    elif bn_type =="dual_bn":
        return(nets.NeuralNetDualNorm, nets.DualParaNet)
    else:
        raise NotImplementedError