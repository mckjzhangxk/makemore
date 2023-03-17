import torch

# 创建一个3x3的张量
input =torch.randn((1,8,7))

# 创建一个形状为(2, 3)的张量，作为要添加的张量
source = torch.randn((5,8,7))

# 创建一个指定索引的LongTensor类型的张量
index = torch.tensor([0,0,0,0,0])

# 在dim=0上根据index张量的指定索引累加source张量
input.index_add_(0, index, source)

# 输出累加后的结果
print(input.shape)