import torch 
from torch import nn 
import math 
from config import *

class TimePositionEmbedding(nn.Module):
    def __init__(self,emb_size):
        super().__init__()
        self.half_emb_size=emb_size//2
        half_emb=torch.exp(torch.arange(self.half_emb_size)*(-1*math.log(10000)/(self.half_emb_size-1)))
        #创建一个从 0 到 half_emb_size-1 的序列 然后乘上-1*math.log(10000)/(self.half_emb_size-1) 这个常数
        self.register_buffer('half_emb',half_emb)

    def forward(self,t):
        t=t.view(t.size(0),1) #把时间扩展为 [batch_size,1] 作为一个列向量
        half_emb=self.half_emb.unsqueeze(0).expand(t.size(0),self.half_emb_size)
        #unsqueeze(0) 将 half_emb 的维度扩展为 [1, half_emb_size]，然后 expand 使其变为 [batch_size, half_emb_size]
        # expand 只能改变维大小为1的维，否则就会报错。不改变的维可以传入-1或者原来的数值。
        #unsqueeze不涉及填充任何数字，它只是改变张量的形状；而expand则会复制张量中的元素以填充到新的形状中
        half_emb_t=half_emb*t
        embs_t = torch.cat((half_emb_t.sin(), half_emb_t.cos()), dim=-1) #(2,8)
        return embs_t
'''
    half_emb*t  采用广播机制 eg. t : [100,200] 扩展完  t: [[100],[200]]
    half_emb: [e^(0*(-1*math.log(10000)/3)),e^(1*(-1*math.log(10000)/3)),e^(2*(-1*math.log(10000)/3)),e^(3*(-1*math.log(10000)/3))]
    扩展完: 
    [[e^(0*(-1*math.log(10000)/3)),e^(1*(-1*math.log(10000)/3)),e^(2*(-1*math.log(10000)/3)),e^(3*(-1*math.log(10000)/3))],
     [e^(0*(-1*math.log(10000)/3)),e^(1*(-1*math.log(10000)/3)),e^(2*(-1*math.log(10000)/3)),e^(3*(-1*math.log(10000)/3))]
    ]
    做乘法的时候广播t成 [[100,100,100,100],[200,200,200,200]] 和half_emd 对应位置做乘法
'''
if __name__=='__main__':
    time_pos_emb=TimePositionEmbedding(8).to(DEVICE)
    t=torch.randint(0,T,(2,)).to(DEVICE)   # 随机2个图片的time时刻
    embs_t=time_pos_emb(t) 
    print(embs_t)