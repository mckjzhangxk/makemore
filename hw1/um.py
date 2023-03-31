import torch
from PIL import Image
import numpy  as np
import random

# 读取图像
img = Image.open('1.jpeg')


# 将 PIL 图像转换为 numpy 数组


# 将 numpy 数组转换为 Tensor
img_tensor = torch.from_numpy( np.array(Image.open('1.jpeg'))).view(-1).float()

x=torch.stack([img_tensor,img_tensor,img_tensor])
# 打印图像 Tensor 的大小
print(img_tensor.size())

# x.transpose_(0,1)
print(x.shape)


x=torch.zeros(3,2,10)
one_hot = torch.nn.functional.one_hot(torch.tensor(3), num_classes=10)

x[0]=one_hot
x[1]=one_hot
# 打印 one-hot 编码
print(one_hot)
print(x)

A=torch.rand(32,3,5,784)
A.transpose_(1,2)
print(A.shape)

x = list(range(11))

# 从列表中无放回地随机取出 5 个元素
selected_elements = random.sample(x, 5)
print(selected_elements)