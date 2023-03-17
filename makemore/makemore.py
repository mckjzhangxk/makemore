import random

import torch
# char level model

with open('name.txt') as fs:
    # s=fs.read()
    # words=s.splitlines()
    words=fs.readlines()
    words=[word[:-1] for word in words] #去掉\n
random.shuffle(words)
chs=sorted(set(''.join(words)))

stoi={c:i+1 for i,c in enumerate(chs)}
stoi['.']=0
itos={i:c for c,i in stoi.items()}


# Bigram Model
PROB=torch.zeros((len(chs)+1,len(chs)+1),dtype=torch.float64)

for word in words:
    for (ch1,ch2) in zip('.'+word,word+'.'):
        idx1,idx2=stoi[ch1],stoi[ch2]
        PROB[idx1,idx2]+=1
PROB/=PROB.sum(dim=1,keepdim=True)


# 创建一个 generator 对象
generator = torch.Generator()
# 设置 generator 的种子
generator.manual_seed(2147483647)
for i in range(10):
    idx=0
    sampleName=''
    while True:
        # 参数不一样，结果都不一样，replacement=True,replacement=False ,sample=1
        idx=torch.multinomial(PROB[idx],1,replacement=True,generator=generator).item()
        sampleName+=itos[idx]
        if idx==0:break
    print(sampleName)


nll=0
cnt=0
for word in words[:1000]:
    for (ch1,ch2) in zip('.'+word,word+'.'):
        idx1,idx2=stoi[ch1],stoi[ch2]
        nll+=-torch.log(PROB[idx1,idx2]).item()
        cnt+=1
print('targit  nll',nll/cnt)
### 使用神经网络


VSize=len(chs)+1
# W => log count
W=torch.randn((VSize,VSize),generator=generator,requires_grad=True)

# 准备训练数据
xs,ys=[],[]
for word in words:
    for (ch1,ch2) in zip('.'+word,word+'.'):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])

xs=torch.nn.functional.one_hot(torch.tensor(xs),num_classes=VSize).float()
ys=torch.tensor(ys)
idx=torch.arange(0,len(ys),dtype=torch.int64)



for k in range(500):
    # forward network
    logit=xs@W
    count=torch.exp(logit)  #logit.exp()
    probs=count/count.sum(dim=1,keepdim=True)
    prob_target=probs[idx,ys]
    nll=-torch.log(prob_target).mean()
    # nll=-(  logit-torch.log (torch.exp(logit).sum(dim=1,keepdim=True)) ).mean()
    if k% 500==0:
        print(nll.item())
    # W.grad.zero_()
    W.grad=None
    nll.backward()
    W.data-=50*W.grad
# sample from neural
for i in range(5):
    idx=0
    sampleName=''
    while True:
        logit=W[idx,:]
        count=logit.exp()
        probs=count/count.sum()

        idx=torch.multinomial(probs,1,replacement=True,generator=generator).item()
        sampleName+=itos[idx]
        if idx==0:break
    print(sampleName)