import torch
import torch.nn as nn
import torch.autograd as autograd

# self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
# f=nn.LSTM(3,22,1,batch_first=True)
B,I,T=32,3,8
device='cuda'
x=torch.rand(B,T,I).to(device)
print(x)
# y,c=f(x)

# print(y.shape)
# print(c[0].shape)  #h_final
# assert tuple(c[1].shape)==(1,B,22) #c_final

# print(torch.cuda.is_available())

# x=torch.tensor([1,2,3,4],dtype=torch.float32,requires_grad=True)
# y=torch.clone(x)

# y[0]=1


# x=[12,2,3]
# y=['a','b','c']


# for k,(c1,c2) in enumerate(zip(x,y)):
#     print(k,c1,c2)
# w=(y*y).sum()
# w.backward()
# print(x.grad)
# print(x)
# g=autograd.grad(w,{"x":x})
# print(g)

# x={
#     "a":1,
#     "b":-1,
#     "c":0,
# }
# print(x.keys())
# print(x.values())
# z=g[0].sum()
# g1=autograd.grad(z,x,allow_unused=True)
# print(g1)
# # print(w)


# x=torch.rand(2,2)


# u=nn.functional.batch_norm(x,None,None,training=True)


# m=x.mean(dim=0)
# s=x.std(dim=0)
# # print(u)
# print(u.mean(0))
# print(u.std(0))
# # print((x-m)/s)
# print(2**0.5)