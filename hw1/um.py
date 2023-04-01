import torch


x=torch.tensor([0,0,0,1,1,1])

x=x.view(2,3)
print(x)

# x=torch.permute(x,(1,0))
x.transpose_(1,0)
print(x)

print(torch.cuda.is_available())