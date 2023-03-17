### bug总结

* Value的 grad 需要更新操作而不是赋值操作
```python
    self.grad+=localfactor * out.grad
```
* 中间节点（运算产生临时节点）是每一次forward都是一个新对象，也就是说只能 forward，backward,不可用backward多次，否则由于zero_grad只作用于参数节点，造成中间节点的grad叠加
* tanh 是往往由于初始化不当，造成 反向grad过小，无法继续训练,relu初始化不当直接截断梯度的传导。