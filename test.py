import torch

# x = torch.randn((2, 2, 3), requires_grad=True)
# y = torch.randn((2, 2, 3), requires_grad=True)
#
# # def distance(vector1,vector2):
# #     return torch.sqrt(torch.square(vector2-vector1).sum())  # pow()是自带函数
# def function(x):
#     theta = torch.randn(x.shape)
#     soft = torch.sign(x) * torch.max(torch.abs(x) - theta, torch.zeros(x.shape))
#     return soft.shape

with open("./output/logs.txt", 'w') as f:
    f.write("123\n")

