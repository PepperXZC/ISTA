import torch

x = torch.randn((2, 2, 3), requires_grad=True)
y = torch.randn((2, 2, 3), requires_grad=True)

def distance(vector1,vector2):
    return torch.sqrt(torch.square(vector2-vector1).sum())  # pow()是自带函数

print(distance(x, y))

