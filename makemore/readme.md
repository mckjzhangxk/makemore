NLP bug
* 在给token建立索引的时候，需要对token进行排序，这样建立的索引才是确定的（set遍历顺序不确定）。
```
    chs=sorted(set(''.join(words)))
    stoi={c:i+1 for i,c in enumerate(chs)}
    stoi['.']=0
    itos={i:c for c,i in stoi.items()}
```

* 一个variable 需要使用data属性进行inplace的更新
```buildoutcfg
W.data-=1*W.grad
```

* 模型不收敛往往是由于 表达式写的不对
```python
logit=xs@W
count=torch.exp(logit)
prob=count/count.sum(dim=1,keepdim=True)
# 忘记的获取目标的prob
nll=-torch.log(prob).mean()

===>
prob=count/count.sum(dim=1,keepdim=True)
prob_target=probs[torch.arange(0,len(ys)),ys]
nll=-torch.log(prob_target).mean()
```

[中文语料](https://github.com/InsaneLife/ChineseNLPCorpus)


* unbind方法的作用

```python
import torch

A = torch.randn(3,4,5)
List=torch.unbind(A,dim=1)

assert List[0]==A[:,0,:]
assert List[1]==A[:,1,:]
assert List[2]==A[:,2,:]
assert List[3]==A[:,3,:]

assert torch.cat(List,dim=1).shape==(3,20)
```

* 自己写的loss func
```python
import torch
 
logit=torch.randn(3,2)
YS=torch.tensor([0,1,2])

count=logit.exp()
probs=count/count.sum(dim=1,keepdim=True)

# 千万不要忘了 torch.arrange
nll=-probs.log()[torch.arange(3),YS].mean()
```
* 决定learning的方法

```python
import torch
steps=1000
lri = torch.linspace(-3,0,steps)
lrs=10**lri

# 统计出loss与lr之间的关系
for step in range(steps):
    lr=lrs[step]
    loss=...
    
```

BUG:
```python
    当step*bacthSize>len(TY)后，永远都是从[0,batch]的数据被训练
    offset=step*bacthSize if step*bacthSize<len(TY) else 0
    xs_batch=TX[offset:offset+bacthSize]
    ys_batch=TY[offset:offset+bacthSize]
```


BUG 封装线性模型的时候，导致没有训练现象模型
```python

    def parameters(self):
        # return [self.w]+[self.b] if self.b is not None else []
        
        return [self.w]+[self.b] if self.b is not None else []
    
    def parameters(self):
        return [self.w,self.b] if self.b is not None else [self.w]
```


```
wavenet总结

A.局部的receive-filed=K+(K-1)*(dilation-1)
B.输出的长度会相对输入减少 receive-filed-1,也就是 (k-1)*(dialation-1)
C. 全局R=(r) + (r{-1}-1) + (r{-2}-1) + (r{-3}-1)+ (r{-4}-1)+ …….
```