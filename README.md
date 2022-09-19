# Transformer
Vit-TransUnet-etc.

1. Vit ==========================================================

如图给定一个input矩阵，转成相同维度的key和value。query经过tanspose后和key相乘，对相乘后结果的最后一个维度做了softmax，可以组成L*L的attention map。map里面每一行的元素总和为1（L×L的方块中，色块越深值越大，色块越浅值越小）。

得到Query，Key和Value都是普通的1x1卷积，差别只在于输出通道大小不同；将Query的输出转置，并和Key的输出相乘，再经过softmax归一化得到一个attention map； 将得到的attention map和Value逐像素点相乘，得到自适应注意力的特征图。

eg: 第一行中，第2个元素对第1个元素比较重要，第3个元素对第1个元素不重要，以此类推，第6和8个元素对它比较重要，其他元素不太重要。这么做的目的是让模型学到输入序列元素之间的重要程度。

![Self-attention](https://user-images.githubusercontent.com/52816016/190941223-beb4966b-bfd1-4ef5-8972-37087bad6601.jpg)

首先，需要把图片输入进网络，和传统的卷积神经网络输入图片不同的是，这里的图片需要分为一个个patch，如图中就是分成了9个patch。每个patch的大小是可以指定的，比如1 16×16等等。然后把每个patch输入到embedding成，也就是Linear Projection of Flattened Patches，通过该层以后，可以得到一系列向量（token），9个patch都会得到它们对应的向量，然后在所有的向量之前加入一个用于分类的向量*，它的维度和其他9个向量一致。此外，还需要加入位置信息，也就是图中所示的0~9。然后把所有的token输入Transformer Encoder中，然后把TransFormer Encoder重复堆叠L次，再将用于分类的token的输出输入MLP Head，然后得到最终分类的结果。

![ViT](https://user-images.githubusercontent.com/52816016/190941236-4bc1c167-5778-4fca-b190-4a3fea8c994b.jpg)

如模型细节图（左）所示，首先输入一张224×224×3的RBG彩色图片，通过Embedding层，具体来说，是由一个16*16大小的卷积核、步距为16的卷积层，得到14*14*768的特征图，然后在高度和宽度（前两个维度）打平，得到196*768的特征向量。然后Concat一个Class token变成尺寸197*768，然后加上Position Embedding，因为尺寸完全相同，所以这里可以理解为数值上的相加，此后经过一个Dropout层，经过重复L次的TransFormer Encoder层，然后跟一个LayerNorm，提取Class token的输出（使用切片操作，将Class信息单独抽出来），紧接着通过MLP Head层得到最终的输出。Pre-Logits是由全连接层+tanh激活函数构成，然后通过一个全连接层得到输出。

![Vit-B16](https://user-images.githubusercontent.com/52816016/190941239-a42c7adc-01e8-4ea1-b579-093a23e76f9f.jpg)

2. XXX ==========================================================
