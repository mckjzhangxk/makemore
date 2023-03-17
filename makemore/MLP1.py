import random

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# char level model
with open('name.txt') as fs:
    # s=fs.read()
    # words=s.splitlines()
    words=fs.readlines()
    words=[word[:-1] for word in words] #去掉\n

chs=sorted(set(''.join(words)))

stoi={c:i+1 for i,c in enumerate(chs)}
stoi['.']=0
itos={i:c for c,i in stoi.items()}

def index2Str(l):
    return "".join([itos[id] for id in l])


#
trainSize=int(len(words)*0.9)
devSize=int(len(words)*0.05)
testSize=len(words)-trainSize-devSize

train_words=words[0:trainSize]
dev_words=words[trainSize:trainSize+devSize]
test_words=words[trainSize+devSize:]

# build dataset
contextSize=3
xs=[]
ys=[]
for word in words[:]:
    word=word+"."
    context=[0]*contextSize
    # print(word)
    for c in word:
        idx=stoi[c]
        xs.append(context)
        ys.append(idx)
        # print(index2Str(context),'->',c)
        context=context[1:]+[idx]

#
XS=torch.tensor(xs)
YS=torch.tensor(ys)
#
#
trainSize=int(len(YS)*0.9)
devSize=int(len(YS)*0.05)
testSize=len(YS)-trainSize-devSize
#
#
trainSet=(XS[0:trainSize],YS[0:trainSize])
devSet=(XS[trainSize:trainSize+devSize],YS[trainSize:trainSize+devSize])
testSet=(XS[-testSize:],YS[-testSize:])
#
print("train_size","dev_size","test_size")
print(len(trainSet[0]),len(devSet[0]),len(testSet[0]))

Vsize=len(stoi)
# build emb layer
embSize=32
#
parameters=[]
generater=torch.Generator()
generater.manual_seed(1234)
W=torch.randn((Vsize,embSize),generator=generater)
parameters.append(W)


hiddenSize=256
W1=torch.randn((embSize*contextSize,hiddenSize),generator=generater)*0.01
b1=torch.zeros(hiddenSize)
parameters.extend([W1,b1])


W2=torch.randn((hiddenSize,Vsize),generator=generater)*0.01
b2=torch.zeros(Vsize)
parameters.extend([W2,b2])
#
for p in parameters:
    p.requires_grad=True
# # 高效,数值稳定
criterion = nn.CrossEntropyLoss()
#
#
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

# inputs是 int 数组
def predict(inputs):
    if len(inputs)<contextSize:
        inputs=[0]*(len(inputs)-contextSize)+inputs
    sampleName = ''.join([itos[idx] for idx in inputs])
    inputs=inputs[:contextSize]

    while True:
        xs=torch.tensor(inputs)
        logit=model(xs)

        probs=nn.functional.softmax(logit,dim=1)

        idx=torch.multinomial(probs,1,replacement=True,generator=generater).item()

        sampleName+=itos[idx]
        if idx==0:
            return sampleName
        inputs=inputs[1:]+[idx]
lr_e=torch.linspace(-3,0,1000)
lr=10**lr_e

bacthSize=32

TX,TY=trainSet

steps=int(100*(len(TX)/bacthSize))
# steps=100000

# lri=[]
# losses=[]
o=list(range(len(TX)))
random.shuffle(o)
TX=TX[o]
TY=TY[o]

offset=0
for step in range(steps):

    # bad  overfit quickly,bug
    offset=offset+bacthSize
    if offset>len(TX):offset=0

    xs_batch=TX[offset:offset+bacthSize]
    ys_batch=TY[offset:offset+bacthSize]

    # idx=torch.randint(0,len(TX),(bacthSize,))
    # xs_batch=TX[idx]
    # ys_batch=TY[idx]

    logit=model(xs_batch)

    nll=nn.functional.cross_entropy(logit,ys_batch)

    if( step%2000==0):
        train_loss=nll.item()
        dev_loss=nn.functional.cross_entropy(model(devSet[0]),devSet[1]).item()

        print(f'train_loss {train_loss},dev_loss {dev_loss}')
        print('--------sample begin--------')
        random.seed(986)
        for k in range(10):
            si=random.randint(0,len(test_words))
            sample=test_words[si]

            sample_pred=predict([stoi[c] for c in sample[:contextSize]])
            print(f'\t{sample}-->{sample_pred}')
        print('--------sample end--------')
    # zero grad
    for p in parameters:
        p.grad=None

    nll.backward()

    for p in parameters:
       p.data-=0.1*p.grad

    # losses.append(nll.item())
    # lri.append(lr_e[step])
# plt.plot(lri,losses)
# plt.show()
# predict

# for i in range(5):
#     # w=predict([stoi['a'],stoi['b']])
#     w=predict([''])
#     print(w)
