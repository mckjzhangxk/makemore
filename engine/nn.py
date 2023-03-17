import random

from engine.autograd import Value

class Module:
    def parameters(self):
        return []
    def zero_grad(self):
        for p in self.parameters():
            p.grad=0

class Neuron(Module):
    def __init__(self,nDims):
       self.w=[Value(random.random(),label=f"w{i}") for i in range(nDims)]
       self.b=Value(0,label='b')

    def __call__(self,Xs):
        return sum([xi*wi for xi,wi in zip(Xs,self.w)])+self.b

    def parameters(self):
        return self.w+[self.b]

class Layer(Module):
    def __init__(self,nDims,oDims,active=False):
        self.C=nDims
        self.F=oDims

        self.active=active
        self.neurons=[Neuron(nDims) for _ in range(oDims)]

    def __call__(self,Xs):
        out=[]
        for n in self.neurons:
            out.append(n(Xs).relu() if self.active else n(Xs))
        return out
    def __repr__(self):
        return f'{self.C}x{self.F},#params={len(self.parameters())}'
    def parameters(self):
        ps=[p for n in self.neurons for p in n.parameters()]
        return  ps

class MLP(Module):
    def __init__(self,C,layers):
        n=len(layers)
        layers=[C]+layers
        self.layers=[Layer(layers[i],layers[i+1],True if  i!=n-1 else False) for i in range(n)]
    def __call__(self, Xs):
        prev=Xs
        for ll in self.layers:
            prev=ll(prev)
        return  prev
    def parameters(self):
        ps=[]
        for ll in self.layers:
            ps.extend(ll.parameters())
        return ps
    def __repr__(self):
        s=''
        for i,ll in enumerate(self.layers):
            s+=f"layer {i}:{ll}\n"
        return s
def forward(model,Xs,Ys):
    Ypred=[]
    for xs in Xs:
        Ypred.append(model(xs)[0])
    losses=[]
    for (yp,ys) in zip(Ypred,Ys):
        lossi=(yp-ys)**2
        losses.append(lossi)
    loss=sum(losses)
    return loss/len(Xs)

Xs=[
    [1,1,3,1],
    [-2,1,-3,2],
    [-1,-1,2,3],
    [2,0,-2,1],
]
Ys=[-1,1,1,-1]


# Xs=[
#     [1],
# ]
# Ys=[-1]
# 问题：
# 1.如下配置后，为什么收敛不了
mlp=MLP(len(Xs[0]),[4,1])

# # solver
#
learning_rate=1e-2
for _ in range(300):
    mlp.zero_grad()
    loss=forward(mlp,Xs,Ys)
    print(loss.data)
    loss.backward()


    for p in mlp.parameters():
        p.data=p.data-learning_rate*p.grad

# for p in mlp.parameters():
#     print(p)
print(mlp(Xs[0]))
print(mlp(Xs[1]))