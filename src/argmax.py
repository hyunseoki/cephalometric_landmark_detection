import torch

def get_gridmap(shape=(10, 10)):
    Ymap, Xmap = np.mgrid[0:shape[1]:1, 0:shape[0]:1]
    Ymap, Xmap = torch.tensor(Ymap.flatten(), dtype=torch.float32).unsqueeze(1),\
                 torch.tensor(Xmap.flatten(),dtype=torch.float32).unsqueeze(1)

    return Ymap, Xmap

def argsoftmax(x, index, beta=1e-5):
    a =torch.exp(-torch.abs(x-x.max())/(beta))
    b =torch.sum(a)
    softmax = a / b
    return torch.sum(softmax * index)

if __name__ == '__main__':
    import numpy as np
    H, W = 10, 10
    Ymap, Xmap = np.mgrid[0:H:1, 0:W:1]
    Ymap, Xmap = torch.tensor(Ymap.flatten(), dtype=torch.float).unsqueeze(1).to('cpu'),\
                 torch.tensor(Xmap.flatten(),dtype=torch.float).unsqueeze(1).to('cpu')

    print('dd')
