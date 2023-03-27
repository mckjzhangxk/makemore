import torch.nn as nn
import torch
import torch.nn.functional as F

class CausalConv1D(nn.Module):
    def __init__(self,inChan,outChan,dilation=1,nonlinearity='linear'):
        super().__init__()
        K=2
        self.receive_field=K+(K-1)*(dilation-1)

        self.pad=nn.ConstantPad1d((self.receive_field-1,0),0)
        self.conv=nn.Conv1d(inChan,outChan,K,dilation=dilation,bias=False)

        nn.init.kaiming_normal_(self.conv.weight,  nonlinearity=nonlinearity)

    def forward(self, x,x_vaid_T=None):
        if x_vaid_T is None:
            x_vaid_T=x.shape[2]

        # x=F.pad(x, (self.receive_field-1,0),mode='constant', value=0)
        x=self.pad(x)
        self.out=self.conv(x)


        out_valid_T=x_vaid_T-(self.receive_field-1)
        return  self.out,out_valid_T

class Block(nn.Module):
    def __init__(self,inChan,dilation=1):
        super(Block, self).__init__()

        self.causul_conv=CausalConv1D(inChan,inChan,dilation,nonlinearity='tanh')  #for tanh
        self.causul_conv2 = CausalConv1D(inChan, inChan, dilation) #for sigma

        self.tanh=nn.Tanh()
        self.sigmoid=nn.Sigmoid()


        self.output_conv=nn.Conv1d(inChan,inChan,1)
        self.skip_conv = nn.Conv1d(inChan, inChan, 1)


    def forward(self, x, x_valid,debug=False):
        h,h_valid=self.causul_conv(x,x_valid)
        h1,h1_valid=self.causul_conv2(x,x_valid)

        th=self.tanh(h)
        sh=self.sigmoid(h1)
        w=th*sh

        o=self.output_conv(w)+x
        s=self.skip_conv(w)

        if debug:
            with torch.no_grad():
                print('pre tanh std', h[-h1_valid:].std().item())
                print('pre sigma std',h1[-h1_valid:].std().item())
                print('tanh std',     th[-h1_valid:].std().item(),'tanh mean',th[-h1_valid:].mean().item())
                print('sigma std',    sh[-h1_valid:].std().item(),'sigma mean',    sh[-h1_valid:].mean().item())
                print('w std',        w[-h1_valid:].std().item())
        return o,s,h_valid



class WaveNet(nn.Module):
    def __init__(self,classes,resChan,stacks,layers):
        super(WaveNet, self).__init__()

        dilations=[2**l for l in range(layers)]*stacks

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

        emb=self.emb_layer(x).transpose(1,2)

        h,Tvalid,=emb,x.shape[1]

        output=0
        for layer in self.layers:
            # bug
            # h,s,Tvalid=layer(emb,Tvalid)

            h, s, Tvalid = layer(h, Tvalid)
            output+=s

        output=output[:,:,-Tvalid:]

        # (B,classes,T)
        logit=self.post_layer(output)

        return logit

