import torch.nn as nn
import torch
import torch.nn.functional as F
from weavenet_fix import CausalConv1D,Block,WaveNet


def testCausalConv1D():
    print("testCausalConv1D")
    B, C, T = 64, 10, 16
    x = torch.randn(B, C, T)
    conv = CausalConv1D(C, 4, dilation=8)
    y = conv(x)

    assert tuple(y.shape) == (B, 4, 8)


    ps=[p for p in conv.parameters()]

    for p in ps:
        print(p.mean(),p.std())
    print('x.std:',x.std())
    # 输入引入了(k-1)*d个0,改变了输入的概率分布，
    # 我把这部分影响去掉了
    print('y.std:',y.std())

def testBlock():
    print("----------------testBlock----------------")

    B,C,T=32,10,16

    xorigin=torch.randn((B,C,T))


    x=xorigin
    block_conv=Block(C,1)
    y,s=block_conv(x)

    assert tuple(y.shape)==(B,C,T-1)
    assert tuple(s.shape)==(B,C,T-1)



    x=xorigin
    block_conv1=Block(C,2)
    y,s=block_conv1(x)

    assert tuple(y.shape)==(B,C,T-2)
    assert tuple(s.shape)==(B,C,T-2)


    x=xorigin
    block_conv2=Block(C,4)
    y,s=block_conv2(x)

    assert tuple(y.shape)==(B,C,T-4)
    assert tuple(s.shape)==(B,C,T-4)



    x=xorigin
    block_conv2=Block(C,8)
    y,s=block_conv2(x)

    assert tuple(y.shape)==(B,C,T-8)
    assert tuple(s.shape)==(B,C,T-8)

def testBlockGrad():
    print("-----------testBlockGrad-----------")
    B,C,T=32,10,16
    x=torch.randn(B,C,T)


    block_conv=Block(C,dilation=1)

    y,s=block_conv(x,debug=True)

    print('x.std', x.std())
    print('y.std', y.std())
    print('s.std', s.std())


def testWavenet():
    print("-------testWavenet---------")
    classes=27
    model=WaveNet(classes,64,1,3)
    params_num=sum(p.numel() for p in model.parameters())
    print(f"#param {params_num}")
    torch.save(model.state_dict(), "wavenet_model.pth")


    B,T=32,8
    x=torch.randint(0,classes,(B,T))
    y = torch.randint(0, classes, size=(B,))

    logits = model(x)
    print(model)
    # for p in model.parameters():
    #     print(tuple(p.data.size()))
    for i in range(500):
        logits=model(x).squeeze()
        loss=F.cross_entropy(logits,y)
        print(loss)
        for p in model.parameters():
            p.grad=None
        loss.backward()
        for p in model.parameters():
            # 最上层的那个y,没有参与loss计算，所以p.grad=None
            if p.grad==None:
                continue
            p.data-=0.5*p.grad


    assert tuple(logits.shape)==(B,classes),f"logit.shape={tuple(logits.shape)}"

    logits=torch.squeeze(logits)

    loss=F.cross_entropy(logits,y)
    print(loss.item())

testCausalConv1D()
testBlock()
testBlockGrad()
testWavenet()
