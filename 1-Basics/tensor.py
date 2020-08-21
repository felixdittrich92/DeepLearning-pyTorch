import torch
x = torch.Tensor(5,3).cuda(device=0) # run on gpu
# Zeile, Spalte
print(type(x))
print(x.shape)
print(x)

# Random tensor
y = torch.randn(5,3).cuda(device=0) 
print(type(y))
print(y.size())
print(y)

# Addition
print(torch.add(x,y))


