import json
import math
class Value:
    def __init__(self,data,label='',children=()):
        self.data=data
        self.label=label
        self.children=children
        self.grad=0
        self._backward = lambda: None
    def toObj(self):
        children = [c.toObj() for c in self.children]
        children=[]
        return  {"data":self.data,"grad":self.grad,"label":self.label,"child":children}
    def __repr__(self):

        o=self.toObj()
        return json.dumps(o)
    def __add__(self, other):
        if isinstance(other, Value):
            out=Value(self.data+other.data,label='+',children=(self,other))
        else:
            out=Value(self.data+other,label='+',children=(self,))

        def backward():
            self.grad+=out.grad
            if isinstance(other, Value):
                other.grad+=out.grad
        out._backward = backward
        return out
    def __radd__(self, other):
        return self+other
    def __sub__(self, other):
        return self+(-1*other)
    def __neg__(self):
        return -1*self
    def __mul__(self, other):
        if isinstance(other, Value):
            out=Value(self.data*other.data,label='*',children=(self,other))
        else:
            out=Value(self.data*other,label='*',children=(self,))

        def backward():
            if isinstance(other, Value):
                self.grad+= other.data * out.grad
                other.grad+=self.data*out.grad
            else:
                self.grad+=other*out.grad
        out._backward=backward
        return out
    def __rmul__(self, other):
        return self*other
    def __truediv__(self, other):
        return self* other**-1
    def __rtruediv__(self, other):
        return other* self**-1
    def __pow__(self, power, modulo=None):
        assert isinstance(power,(int,float)), "only supporting int/float powers for now"
        out=Value(self.data**power,label='pow',children=(self,))

        def backward():
            self.grad+=power*self.data**(power-1)*out.grad
        out._backward=backward
        return  out

    def tanh(self):
        t=math.tanh(self.data)
        out=Value(t,label='tanh',children=(self,))
        def backward():
            self.grad+=1-t*t
        out._backward=backward
        return out
    def relu(self):
        out=Value(self.data if self.data>0 else 0,children=(self,))
        def backward():
            self.grad+=(out.data>0)*out.grad
        out._backward=backward
        return out
    def exp(self):
        out=Value(math.exp(self.data),label='exp',children=(self,))
        def backward():
            self.grad+=math.exp(self.data)*out.grad
        out._backward=backward
        return out
    def backward(self):

        visited=[]
        def toplogical(root):
            for ch in root.children:
                if ch in visited:
                    continue
                toplogical(ch)
            visited.append(root)

        toplogical(self)
        self.grad=1
        for n in reversed(visited):
            n._backward()