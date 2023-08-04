import torch
import random
import numpy as np
import os
import torch.backends.cudnn as cudnn
from datetime import timedelta

# def set_seeds(seed=0, cuda_deterministic=True):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
#     if cuda_deterministic:  # slower, more reproducible
#         cudnn.deterministic = True
#         cudnn.benchmark = False
#     else:  # faster, less reproducible
#         cudnn.deterministic = False
#         cudnn.benchmark = True

# set requies_grad=Fasle to avoid computation
def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def dist_init(rank, world_size, temp_dir):
    # Init torch.distributed.
    # if world_size > 1:
        # set up the master's ip address so this child process can coordinate
        # MASTER_ADDR = '127.0.0.1'
        # os.environ['MASTER_ADDR'] = MASTER_ADDR
        # MASTER_PORT = find_free_port()
        # os.environ['MASTER_PORT'] = MASTER_PORT
    timeout = 6
    if os.name == 'nt':
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        init_method = 'file:///' + init_file.replace('\\', '/')
        torch.distributed.init_process_group(backend='gloo', init_method=init_method,
                                             rank=rank, world_size=world_size,
                                             timeout=timedelta(hours=timeout))
    else:
        # MASTER_PORT = find_free_port()
        # init_method = 'tcp://127.0.0.1:' + MASTER_PORT
        # print('init_method', init_method)
        # exit()
        # torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank,
        #                                      world_size=world_size)
        torch.distributed.init_process_group(backend='nccl', rank=rank,
                                             world_size=world_size,
                                             timeout=timedelta(hours=timeout))
    # Init torch_utils.
    sync_device = torch.device('cuda', rank)
    return sync_device