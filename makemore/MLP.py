import random

import torch
import torch.nn.functional as F


# build dataset
contextSize=3
# char level model

words=open('name.txt').read().splitlines()
import random
random.seed(42)
random.shuffle(words)

chs=sorted(set(''.join(words)))
stoi={c:i+1 for i,c in enumerate(chs)}
stoi['.']=0
itos={i:c for c,i in stoi.items()}

def index2Str(l):
    return "".join([itos[id] for id in l])

def build_dataset(words):
    Xs,Ys=[],[]
    for word in words:
        context = [0] * contextSize
        for ch in word+".":
            idx=stoi[ch]
            Xs.append(context)
            Ys.append(idx)
            context=context[1:]+[idx]
    return torch.tensor(Xs),torch.tensor(Ys)

n1=int(len(words)*0.8)
n2=int(len(words)*0.9)

Xtr,Ytr=build_dataset(words[0:n1])
Xdev,Ydev=build_dataset(words[n1:n2])
Xtest,Ytest=build_dataset(words[n2:])
test_words=words[n2:]


print(Xtr.shape,Ytr.shape)
print(Xdev.shape,Ydev.shape)
print(Xtest.shape,Ytest.shape)



# build emb layer
Vsize=len(stoi)
embSize=10
hiddenSize=200



generater=torch.Generator()
generater.manual_seed(2147483647)
W=torch.randn((Vsize,embSize),generator=generater)

W1=torch.randn((embSize*contextSize,hiddenSize),generator=generater)*(5/3)/(hiddenSize**0.5)
b1=torch.randn((hiddenSize,),generator=generater)

W2=torch.randn((hiddenSize,Vsize),generator=generater)*0.01

# b2=torch.randn((Vsize,),generator=generater)
b2=torch.zeros((Vsize,))


parameters=[W,W1,b1,W2,b2]


for p in parameters:
    p.requires_grad=True


print(f"# {sum([p.nelement() for p in parameters])} paramaters",)

def model(xs):
    # W[XS].shape= XS.shape + W.shape[1:]
    emb = W[xs].view(-1, contextSize * embSize)
    # 以下做法并不高效
    # [(n,embsize),(n,embsize),(n,embsize)]
    # contexvList=torch.unbind(W[XS],dim=1)
    # emb1=torch.cat(contexvList,dim=1)
    # print(emb1.numpy()==emb.numpy())

    L1=(emb@W1+b1).tanh()
    logit=L1@W2+b2

    return logit

# inputs是 int 数组,运行传入空数组
def predict(inputs):
    sampleName = ''.join([itos[idx] for idx in inputs])
    inputs=[0]*contextSize+inputs
    inputs=inputs[-contextSize:]

    while True:
        xs=torch.tensor(inputs)
        logit=model(xs)

        probs=F.softmax(logit,dim=1)

        idx=torch.multinomial(probs,1,replacement=True,generator=generater).item()

        sampleName+=itos[idx]
        if idx==0:
            return sampleName
        inputs=inputs[1:]+[idx]
@torch.no_grad()
def splitLoss(split):
    d={
        'train':(Xtr,Ytr),
        'dev':(Xdev,Ydev),
        'tetst':(Xtest,Ytest)
    }
    xs,ys=d[split]
    loss=F.cross_entropy(model(xs),ys).item()
    return loss

lr=0.1
bacthSize=32

# steps=int(10*(len(Xtr)/bacthSize))
steps=200000
devLoss=[]
offset=0
for step in range(steps):
    idx=torch.randint(0,Xtr.shape[0],(bacthSize,),generator=generater)
    xs_batch=Xtr[idx]
    ys_batch=Ytr[idx]

    # offset=offset+bacthSize if offset+bacthSize<len(Xtr) else 0
    # xs_batch=Xtr[offset:offset+bacthSize]
    # ys_batch=Ytr[offset:offset+bacthSize]

    logit=model(xs_batch)

    nll=F.cross_entropy(logit,ys_batch)


    # zero grad
    for p in parameters:
        p.grad=None

    nll.backward()
    lr = 0.1 if step < 100000 else 0.01
    for p in parameters:
       p.data-=lr*p.grad

    if( step%2000==0):
        train_loss=nll.item()
        dev_loss=splitLoss('dev')
        devLoss.append(dev_loss)
        print(f'{step:7d}/{steps:7d} train_loss {train_loss:.4f},dev_loss {dev_loss:.4f}')
        # print('--------sample begin--------')
        # random.seed(986)
        # for k in range(10):
        #     si=random.randint(0,len(test_words))
        #     sample=test_words[si]
        #
        #     sample_pred=predict([stoi[c] for c in sample[:2]])
        #     print(f'\t{sample}-->{sample_pred}')
        # print('--------sample end--------')



final_train_loss=splitLoss('train')
final_dev_loss=splitLoss('dev')

print(f"minLoss={min(devLoss): .4f} train loss {final_train_loss:.4f}, dev loss {final_dev_loss:.4f}")

# predict

for i in range(0):
    # w=predict([stoi['a'],stoi['b']])
    w=predict([])
    print(w)