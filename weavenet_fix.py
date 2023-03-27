import torch.nn as nn
import torch
import torch.nn.functional as F

class CausalConv1D(nn.Module):
    def __init__(self,inChan,outChan,dilation=1,nonlinearity='linear'):
        super().__init__()
        K=2
        self.receive_field=K+(K-1)*(dilation-1)

        self.conv=nn.Conv1d(inChan,outChan,K,dilation=dilation,bias=False)

        nn.init.kaiming_normal_(self.conv.weight,  nonlinearity=nonlinearity)

    def forward(self, x):
        # x=F.pad(x, (self.receive_field-1,0),mode='constant', value=0)
        self.out=self.conv(x)

        return  self.out

class Block(nn.Module):
    def __init__(self,inChan,dilation=1):
        super(Block, self).__init__()

        self.causul_conv=CausalConv1D(inChan,inChan,dilation,nonlinearity='tanh')  #for tanh
        self.causul_conv2 = CausalConv1D(inChan, inChan, dilation) #for sigma

        self.tanh=nn.Tanh()
        self.sigmoid=nn.Sigmoid()


        self.output_conv=nn.Conv1d(inChan,inChan,1)
        self.skip_conv = nn.Conv1d(inChan, inChan, 1)


    def forward(self, x,debug=False):
        h=self.causul_conv(x)
        h1=self.causul_conv2(x)

        th=self.tanh(h)
        sh=self.sigmoid(h1)
        w=th*sh

        o=self.output_conv(w)+x[:,:,-h.shape[2]:]
        s=self.skip_conv(w)

        if debug:
            with torch.no_grad():
                print()

                print('pre x std', x.std().item())
                print('pre tanh std', h.std().item())
                print('pre sigma std',h1.std().item())
                print('tanh std',     th.std().item(),'tanh mean',th.mean().item())
                print('sigma std',    sh.std().item(),'sigma mean',    sh.mean().item())
                print('tahh*sigma std',        w.std().item())

                print()
        return o,s



class WaveNet(nn.Module):
    def __init__(self,classes,resChan,stacks,layers):
        super(WaveNet, self).__init__()

        dilations=[2**l for l in range(layers)]*stacks

        self.receive_filed=1
        for i,d in enumerate(dilations):
            self.receive_filed+=d


        self.emb_layer=nn.Embedding(classes,resChan)
        self.layers=[Block(resChan,d) for d in dilations]


        self.post_layer=nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(resChan,resChan,1),
            nn.ReLU(),
            nn.Conv1d(resChan, classes, 1),
        )
    def forward(self,x):
        # x.shape =(B,T,classes)
        B,T=x.shape
        remain=T-self.receive_filed+1

        emb=self.emb_layer(x).transpose(1,2)

        h=emb

        output=0
        for layer in self.layers:
            h,s=layer(h)
            output+=s[:,:,-remain:]


        # (B,classes,T)
        logit=self.post_layer(output)

        return logit

