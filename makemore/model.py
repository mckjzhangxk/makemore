
import torch
class LinearLayer():
    def __init__(self,fin,fout,bias=True):

        self.w=torch.randn((fin,fout))
        self.b=torch.zeros((fout,)) if bias else None

    def __call__(self, x):
        self.out=x@self.w
        if self.b is not None:
            self.out+=self.b
        return self.out
    def parameters(self):
        return [self.w]+ [self.b] if self.b is not None else []



X=torch.randn(100,10)
layers=[
    LinearLayer(10,81),
    LinearLayer(81,49),
    LinearLayer(49, 25),
    LinearLayer(25, 225),
]

parameters=[p for layer in layers for p in layer.parameters()]
for p in parameters:
    p.requires_grad=True

h=X
for layer in layers:
    h=layer(h)
    h.retain_grad()

print("forward pass....")
print(f"layer 0,std {X.std().item() :.3f}")
for i,layer in enumerate(layers):
    print(f"layer {i+1},std {layer.out.std().item() :.3f}")


a=torch.randn(225,1)
a.requires_grad=True
a.retain_grad()
L=(h@a).sum()

print("\nbackward pass....")
print(L.item())
L.backward()

print(a.grad.std())


for i,layer in enumerate(reversed(layers)):
    print(f"layer {len(layers)-i},std {layer.out.grad.std():.3f}")


