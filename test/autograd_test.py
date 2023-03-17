from engine.autograd import Value
from engine.nn import *
import numpy as np


def test1():
    a = Value(1)
    b = Value(2)

    c = a + b
    c.grad = 1
    c._backward()

    assert a.grad == c.grad
    assert b.grad==c.grad

    a.grad=0
    b.grad=0

    a2=a+a
    c=a2+a

    c.grad=1
    c._backward()
    a2._backward()

    assert a.grad==3


    a.grad=0
    b.grad=0
    c=a*22
    c.grad=1
    c._backward()
    assert a.grad == 22

    a.grad=0
    b.grad=0

    a3=3*a
    c=a+a3
    c.grad=1
    c._backward()
    a3._backward()
    assert a.grad == 4


    a=Value(4)
    a3p=a**3
    c=a3p+a
    c.grad=1
    c._backward()
    a3p._backward()
    assert a.grad == 49

    a=Value(2)
    b=Value(3)
    #
    c=a+b
    c.backward()
    assert a.grad == c.grad
    assert b.grad==c.grad

    a=Value(2)
    b=Value(3)
    #
    c=a+b*a
    c.backward()
    assert a.grad == 4
    assert b.grad== 2

    a=Value(2)
    b=Value(3)

    c=a+2*b+b**3
    c.backward()
    assert b.grad==29

def test_exp():
    a=Value(3)
    c=a.exp()
    c.backward()
    assert c.data==a.grad

    a = Value(3)
    b= 2*a+3
    c = b.exp()
    c.backward()

    assert 2*c.data == a.grad
def test_div():
    a = Value(3)
    b=  Value(32)

    c=a/b
    c.backward()

    assert a.grad==1/b.data
    assert b.grad==-a.data/b.data/b.data

    c=1/b
    assert c.data==1/b.data



    b=  Value(0.5)
    c=b+1/(b*b)
    c.backward()
    assert b.grad==(1-2/(b.data**3))

def test_tanh():
    x=Value(0.5)
    y=x.tanh()
    y.backward()

    x1 = Value(0.5)
    ex=x1.exp()
    mex=(-x1).exp()

    y1=(ex-mex)/(ex+mex)
    y1.backward()


    np.testing.assert_array_almost_equal(y.data,y1.data)
    np.testing.assert_array_almost_equal(x1.grad,x.grad)
def testNeuron():
    n=Neuron(2)
    o=n([1,0])
    assert o.data==n.w[0].data
    o = n([0, 1])
    assert o.data == n.w[1].data


    assert len(n.parameters())==3

    layer=Layer(3,4)
    out=layer([0,1,2])
    assert len(out)==4

    assert len(layer.parameters())==(12+4)

    out[0].backward()
    layer.zero_grad()

    for p in layer.parameters():
        assert p.grad==0

    layer1=Layer(3,4,True)
    out=layer1([0,1,2])

    layer2 = Layer(4, 1, False)
    out=layer2(out)
    # print(out)

    # MLP
    Xs=[1,2,3]
    mlp=MLP(3,[4,3,5,1])
    assert len(mlp(Xs))==1


    # print(mlp)


def testZeroGrad():
    n=Neuron(1)
    x=[10]

    n.zero_grad()
    l=n(x)
    l.backward()


    p1=[p.grad for p in n.parameters()]

    n.zero_grad()
    l=n(x)
    l.backward()
    p2=[p.grad for p in n.parameters()]
    # print(l)
    for (a,b) in zip(p1,p2):
        # print(a,b)
        assert a==b
test1()
test_exp()
test_div()
test_tanh()
testNeuron()
testNeuron()
testZeroGrad()
