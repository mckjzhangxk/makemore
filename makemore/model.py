import random
import torch
import torch.nn.functional as F

class Module():
    def parameters(self):
        return []
    def zero_grads(self):
        pass
class LinearLayer(Module):
    def __init__(self,fin,fout,bias=True,gain=1,kamInit=True):
        if kamInit:
            g=(fin**-0.5)
        else:
            g=1
        self.w=torch.randn((fin,fout)) *g*gain
        self.b=torch.zeros((fout,)) if bias else None

    def __call__(self, x):
        self.out=x@self.w
        if self.b is not None:
            self.out+=self.b
        return self.out
    def parameters(self):
        return [self.w,self.b] if self.b is not None else [self.w]
    def zero_grads(self):
        self.w.grad=None
        if self.b is not None:
            self.b.grad=None
class Tanh(Module):
    def __call__(self,x):
        self.out=torch.tanh(x)
        return self.out
class BatchNorm1D(Module):
    def __init__(self,C,moment=0.99):
        self.moment=moment
        self.bngain=torch.ones((C,))
        self.bnbias=torch.zeros((C,))
        self.training=True


        self.running_bngain=torch.zeros((C,))
        self.running_bnbias=torch.zeros((C,))
    def __call__(self,x):
        if self.training:
            dim=[i for i in range(len(x.shape)-1)]
            umean=torch.mean(x,dim=dim)
            var=torch.var(x,dim=dim,unbiased=True)

            with torch.no_grad():
                self.running_bngain=self.moment*self.running_bngain+(1-self.moment)*(var**0.5)
                self.running_bnbias=self.moment*self.running_bnbias+(1-self.moment)*umean
            xbar=(x-umean)*( (var+1e-5)**-0.5 )
        else:
            xbar=(x-self.running_bnbias)*(self.running_bngain **-1 )
        self.out=xbar* self.bngain+self.bnbias
        return self.out
    def parameters(self):
        return [self.bngain,self.bnbias]
    def zero_grads(self):
        self.bngain.grad=None
        self.bnbias.grad = None
class Flatten(Module):
    def __call__(self,x):
        self.out=out=x.view(x.shape[0],-1)
        return self.out
class FlattenConsecutive(Module):
    def __init__(self,n):
        self.n=n
    def __call__(self,x):
        self.out=out=x.view(x.shape[0],-1,self.n*x.shape[2])
        if self.out.shape[1]==1:
            self.out=torch.squeeze(self.out)
        return self.out
class EmbedingLayer(Module):
    def __init__(self,n_size,n_embd_size):
        self.weight=torch.randn(n_size,n_embd_size)
    def __call__(self,Xb):
        self.out=self.weight[Xb]
        return  self.out
    def zero_grads(self):
        self.weight.grad=None
    def parameters(self):
        return [self.weight]
class SequenceContainer(Module):
    def __init__(self,layers):
        self.layers=layers
    def __call__(self, x):
        self.out=x
        for layer in self.layers:
            self.out=layer(self.out)
        return self.out
    def zero_grads(self):
        for layer in self.layers:
            layer.zero_grads()
    def train(self):
        for layer in self.layers:
            layer.training=True
            for p in layer.parameters():
                p.requires_grad=True
    def val(self):
        for layer in self.layers:
            layer.training = False
    def parameters(self):
        params=[p for layer in self.layers for p in layer.parameters() ]
        return params

class DataSet():
    def __init__(self):
        self.words = open('name.txt').read().splitlines()
        random.shuffle(self.words)

        self.chs = sorted(set(''.join(self.words)))
        self.stoi = {c: i + 1 for i, c in enumerate(self.chs)}
        self.stoi['.'] = 0
        self.itos = {i: c for c, i in self.stoi.items()}
        self.VSIZE=len(self.itos)
    def index2Str(self,l):
        return "".join([self.itos[id] for id in l])
    def _build_dataset(self,words,contextSize):
        Xs,Ys=[],[]
        for word in words:
            context = [0] * contextSize
            for ch in word+".":
                idx=self.stoi[ch]
                Xs.append(context)
                Ys.append(idx)
                context=context[1:]+[idx]
        return torch.tensor(Xs),torch.tensor(Ys)

    def getData(self,tag,contextSize):
        n1 = int(len(self.words) * 0.8)
        n2 = int(len(self.words) * 0.9)

        if tag=='train':
            X,Y=self._build_dataset(self.words[0:n1],contextSize)
        elif tag=='val':
            X, Y = self._build_dataset(self.words[n1:n2],contextSize)
        elif tag=='test':
            X, Y = self._build_dataset(self.words[n2:],contextSize)

        print(f"{tag}: X: {X.shape}, Y: {Y.shape}")

        return X,Y
@torch.no_grad()
def splitLoss(split):
    d={
        'train':(Xtr,Ytr),
        'dev':(Xdev,Ydev),
        'tetst':(Xtest,Ytest)
    }
    xs,ys=d[split]
    if split=="train":
        model.train()
    else:
        model.val()
    loss=F.cross_entropy(model(xs),ys).item()
    return loss

hparams={
        "contextSize":8,
        "embSize":16,
        "hiddenSize":256,
        "steps":200000,
        "batch_size":64,
        "Wgain":5/3,
        "softmax_gain":0.01
}

random.seed(42)
torch.manual_seed(2147483647)


ds=DataSet()
Xtr,Ytr=ds.getData('train',hparams["contextSize"])
Xdev,Ydev=ds.getData('val',hparams["contextSize"])
Xtest,Ytest=ds.getData('test',hparams["contextSize"])


embSize=hparams["embSize"]
hiddenSize=hparams["hiddenSize"]
Wgain=hparams["Wgain"]
softmax_gain=hparams["softmax_gain"]

models=[
    SequenceContainer([
        EmbedingLayer(ds.VSIZE,embSize),
        Flatten(),
        LinearLayer(embSize*hparams["contextSize"],hiddenSize,True,Wgain,False), Tanh(),
        LinearLayer(hiddenSize, ds.VSIZE,   True, softmax_gain,False),
    ]),
    SequenceContainer([
        EmbedingLayer(ds.VSIZE, embSize),
        Flatten(),
        LinearLayer(embSize * hparams["contextSize"], hiddenSize, True, Wgain), Tanh(),
        LinearLayer(hiddenSize, ds.VSIZE, True, softmax_gain),
    ]),
    SequenceContainer([
        EmbedingLayer(ds.VSIZE,embSize),
        Flatten(),
        LinearLayer(embSize*hparams["contextSize"],hiddenSize,False,Wgain),  BatchNorm1D(hiddenSize), Tanh(),
        LinearLayer(hiddenSize,hiddenSize,  False, Wgain),BatchNorm1D(hiddenSize), Tanh(),
        LinearLayer(hiddenSize, hiddenSize, False, Wgain), BatchNorm1D(hiddenSize), Tanh(),
        LinearLayer(hiddenSize, hiddenSize, False, Wgain), BatchNorm1D(hiddenSize), Tanh(),
        LinearLayer(hiddenSize, hiddenSize, False, Wgain), BatchNorm1D(hiddenSize), Tanh(),
        LinearLayer(hiddenSize, ds.VSIZE,   False, softmax_gain),BatchNorm1D(ds.VSIZE)
    ]),
    SequenceContainer([
        EmbedingLayer(ds.VSIZE, embSize),
        Flatten(),
        LinearLayer(embSize * hparams["contextSize"], hiddenSize, True, Wgain), Tanh(),
        LinearLayer(hiddenSize, hiddenSize, True, Wgain), Tanh(),
        LinearLayer(hiddenSize, hiddenSize, True, Wgain),Tanh(),
        LinearLayer(hiddenSize, hiddenSize, True, Wgain),Tanh(),
        LinearLayer(hiddenSize, hiddenSize, True, Wgain), Tanh(),
        LinearLayer(hiddenSize, ds.VSIZE, False, softmax_gain)
    ]),
    SequenceContainer([
        EmbedingLayer(ds.VSIZE,embSize),
        FlattenConsecutive(2),
        LinearLayer(2*embSize,hiddenSize,bias=False),BatchNorm1D(hiddenSize),Tanh(),
        FlattenConsecutive(2),
        LinearLayer(2 * hiddenSize, hiddenSize, bias=False), BatchNorm1D(hiddenSize), Tanh(),
        FlattenConsecutive(2),
        LinearLayer(2*hiddenSize, ds.VSIZE, gain=hparams["softmax_gain"], bias=False), BatchNorm1D(ds.VSIZE)
    ])

]
model=models[-1]

# model=SequenceContainer([
#     EmbedingLayer(ds.VSIZE,embSize),
#     Flatten(),
#     LinearLayer(embSize*hparams["contextSize"],hiddenSize,False,Wgain),  BatchNorm1D(hiddenSize), Tanh(),
#     LinearLayer(hiddenSize,hiddenSize,  False, Wgain),BatchNorm1D(hiddenSize), Tanh(),
#     LinearLayer(hiddenSize, hiddenSize, False, Wgain), BatchNorm1D(hiddenSize), Tanh(),
#     LinearLayer(hiddenSize, hiddenSize, False, Wgain), BatchNorm1D(hiddenSize), Tanh(),
#     LinearLayer(hiddenSize, hiddenSize, False, Wgain), BatchNorm1D(hiddenSize), Tanh(),
#     LinearLayer(hiddenSize, ds.VSIZE,   False, softmax_gain),BatchNorm1D(ds.VSIZE)
# ])

ncount=sum([p.nelement() for p in model.parameters()])
print(f"# {ncount}")


#
# model=SequenceContainer([
#     EmbedingLayer(ds.VSIZE,embSize),
#     FlattenCon(2),
#     LinearLayer(2*embSize,hiddenSize,bias=False),BatchNorm1D(hiddenSize),Tanh(),
#     FlattenCon(2),
#     LinearLayer(2 * hiddenSize, hiddenSize, bias=False), BatchNorm1D(hiddenSize), Tanh(),
#     FlattenCon(2),
#     LinearLayer(2*hiddenSize, ds.VSIZE, gain=hparams["softmax_gain"], bias=False), BatchNorm1D(ds.VSIZE)
# ])
print("--------------training---------------")
print(f"hparams:{hparams}")
print()

model.train()
parameters=model.parameters()
for step in range(hparams["steps"]):
    idx=torch.randint(0,Xtr.shape[0],(hparams["batch_size"],),generator=None)
    xs_batch=Xtr[idx]
    ys_batch=Ytr[idx]


    logit=model(xs_batch)
    nll=F.cross_entropy(logit,ys_batch)

    model.zero_grads()
    nll.backward()

    lr = 0.1 if step < 100000 else 0.01
    for p in parameters:
       p.data-=lr*p.grad
    if( step%2000==0):
        train_loss=nll.item()
        dev_loss=splitLoss('dev')
        print(f'{step:7d}/{hparams["steps"]:7d} train_loss {train_loss:.4f},dev_loss {dev_loss:.4f}')
        model.train()

print("--------------training end---------------")


final_train_loss=splitLoss('train')
final_dev_loss=splitLoss('dev')
print(f"train loss {final_train_loss:.4f}, dev loss {final_dev_loss:.4f}")
# #
# x=torch.randint(0,ds.VSIZE,(128,3))
# y=model(x)
#
# for layer in model.layers:
#     print(f"{type(layer).__name__}: shape:{layer.out.shape} ,mean:{torch.mean(layer.out).item() :.2f},std:{torch.std(layer.out).item() :.2f}")



# 以下是理解 放大缩小 输出与grad的一段代码
# X=torch.randn(100,10)
# layers=[
#     LinearLayer(10,81),
#     LinearLayer(81,49),
#     LinearLayer(49, 25),
#     LinearLayer(25, 225),
# ]
#
# parameters=[p for layer in layers for p in layer.parameters()]
# for p in parameters:
#     p.requires_grad=True
#
# h=X
# for layer in layers:
#     h=layer(h)
#     h.retain_grad()
#
# print("forward pass....")
# print(f"layer 0,std {X.std().item() :.3f}")
# for i,layer in enumerate(layers):
#     print(f"layer {i+1},std {layer.out.std().item() :.3f}")
#
#
# a=torch.randn(225,1)
# a.requires_grad=True
# a.retain_grad()
# L=(h@a).sum()
#
# print("\nbackward pass....")
# print(L.item())
# L.backward()
#
# print(a.grad.std())


# for i,layer in enumerate(reversed(layers)):
#     print(f"layer {len(layers)-i},std {layer.out.grad.std():.3f}")


