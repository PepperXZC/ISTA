import torch

x = torch.ones((2,2), requires_grad=False)
y = x * x

y.sum().backward()
print(x.grad)

