# test if gpu works in pytorch

import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available. Using GPU.")