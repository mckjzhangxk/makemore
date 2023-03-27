import random
import torch
import torch.nn.functional as F
from wavenet.weavenet import WaveNet


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
        model.eval()
    loss=F.cross_entropy(model(xs),ys).item()
    return loss

hparams={
        "contextSize":8,
        "embSize":16,
        "hiddenSize":100,
        "steps":200000,
        "batch_size":32,
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
    WaveNet(ds.VSIZE,hiddenSize,1,3)
]
model=models[-1]

def zeros_grad(model):
    for p in model.parameters():
       p.grad=None



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

xs_batch = torch.randint(0, ds.VSIZE, (hparams["batch_size"], 8))
ys_batch = torch.randint(0, ds.VSIZE, size=(hparams["batch_size"],))

for step in range(hparams["steps"]):
    # torch.manual_seed(2147483647)
    # idx=torch.randint(0,Xtr.shape[0],(hparams["batch_size"],),generator=None)
    # xs_batch=Xtr[idx]
    # ys_batch=Ytr[idx]


    logit=model(xs_batch)
    nll=F.cross_entropy(logit,ys_batch)

    # model.zero_grads()
    # zeros_grad(model)
    for p in model.parameters():
        p.grad = None

    nll.backward()

    lr = 1 if step < 100000 else 0.01
    for p in parameters:
       p.data-=lr*p.grad

    if( step%2==0):
        train_loss=nll.item()
        # dev_loss=splitLoss('dev')
        print(f'{step:7d}/{hparams["steps"]:7d} train_loss {train_loss:.4f},dev_loss {-1:.4f}')
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


