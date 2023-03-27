import torch.nn as nn
import torch
import torch.nn.functional as F
from weavenet import CausalConv1D,Block,WaveNet


def testCausalConv1D():
    B, C, T = 64, 10, 16
    x = torch.randn(B, C, T)
    conv = CausalConv1D(C, 4, dilation=8)
    y, yT = conv(x)

    assert tuple(y.shape) == (B, 4, T)
    assert yT == 8


    ps=[p for p in conv.parameters()]

    for p in ps:
        print(p.mean(),p.std())
    print('x.std:',x.std())
    # 输入引入了(k-1)*d个0,改变了输入的概率分布，
    # 我把这部分影响去掉了
    print('y.std:',y[:,:,-yT:].std())

def testBlock():

    B,C,T=32,10,16
    x=torch.randn(B,C,T)
    conv=CausalConv1D(C,4,dilation=8)
    y,yT=conv(x)

    assert  tuple(y.shape)==(B,4,T)
    assert yT==8


    block_conv=Block(C,1)
    y,s,yT=block_conv(x,T)

    assert tuple(y.shape)==(B,C,T)
    assert tuple(s.shape)==(B,C,T)

    assert yT==T-1

    x,xT=y,yT
    block_conv1=Block(C,2)
    y,s,yT=block_conv1(x,xT)

    assert tuple(y.shape)==(B,C,T)
    assert tuple(s.shape)==(B,C,T)
    assert yT==xT-2


    x,xT=y,yT
    block_conv2=Block(C,4)
    y,s,yT=block_conv2(x,xT)

    assert tuple(y.shape)==(B,C,T)
    assert tuple(s.shape)==(B,C,T)
    assert yT==xT-4


    x,xT=y,yT
    block_conv2=Block(C,8)
    y,s,yT=block_conv2(x,xT)

    assert tuple(y.shape)==(B,C,T)
    assert tuple(s.shape)==(B,C,T)
    assert yT==xT-8

def testBlockGrad():
    print("-----------testBlockGrad-----------")
    B,C,T=32,10,16
    x=torch.randn(B,C,T)


    block_conv=Block(C,dilation=1)

    y,s,yt=block_conv(x,T,debug=True)

    print('x.std', x.std())
    print('y.std', y[-yt:].std())
    print('s.std', s[-yt:].std())


def testWavenet():
    classes=27
    model=WaveNet(classes,100,1,4)
    params_num=sum(p.numel() for p in model.parameters())
    print(f"#param {params_num}")
    torch.save(model.state_dict(), "wavenet_model.pth")


    B,T=32,16
    x=torch.randint(0,classes,(B,T))
    y = torch.randint(0, classes, size=(B,))

    logits = model(x)
    for i in range(1000):
        logits=model(x).squeeze()
        loss=F.cross_entropy(logits,y)
        print(loss)
        for p in model.parameters():
            p.grad=None
        loss.backward()
        for p in model.parameters():
            p.data-=1*p.grad

    assert tuple(logits.shape)==(B,classes,1),f"logit.shape={tuple(logits.shape)}"

    logits=torch.squeeze(logits)

    loss=F.cross_entropy(logits,y)
    print(loss.item())

testCausalConv1D()
testBlock()
testBlockGrad()
testWavenet()
