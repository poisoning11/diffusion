import torch 
from torch import nn 
from dataset import train_dataset
from config import * 
from diffusion import forward_diffusion
from time_position_emb import TimePositionEmbedding
from conv_block import ConvBlock

class UNet(nn.Module):
    def __init__(self,img_channel,channels=[64, 128, 256, 512, 1024],time_emb_size=256,qsize=16,vsize=16,fsize=32,cls_emb_size=32):
        super().__init__()

        channels=[img_channel]+channels
        
        # time转embedding
        self.time_emb=nn.Sequential(
            TimePositionEmbedding(time_emb_size),
            nn.Linear(time_emb_size,time_emb_size),
            nn.ReLU(),
        )

        # 引导词cls转embedding
        self.cls_emb=nn.Embedding(10,cls_emb_size)
        '''
            
        在神经网络中，nn.Embedding层是可训练的。它通过训练来学习输入数据中的语义或结构信息，使得相似的输入在嵌入空间中具有相似的向量表示
        。nn.Embedding层中的权重（即嵌入矩阵）被视为模型的可学习参数，在训练过程中，模型会根据损失函数调整嵌入矩阵，使其更好地表示词语的语义特征
        。初始化时，nn.Embedding层的权重是随机生成的，但随着模型的训练，这些权重会通过反向传播和梯度下降算法进行更新
        。因此，nn.Embedding层的参数会参与梯度下降，也就是更新模型参数的过程中也会更新nn.Embedding的参数
        。简而言之，nn.Embedding层在构建模型过程中可以作为模型的一部分，进行共同训练
        。
        '''
        # 每个encoder conv block增加一倍通道数
        self.enc_convs=nn.ModuleList()
        for i in range(len(channels)-1):
            self.enc_convs.append(ConvBlock(channels[i],channels[i+1],time_emb_size,qsize,vsize,fsize,cls_emb_size))
        
        # 每个encoder conv后马上缩小一倍图像尺寸,最后一个conv后不缩小
        self.maxpools=nn.ModuleList()
        for i in range(len(channels)-2):
            self.maxpools.append(nn.MaxPool2d(kernel_size=2,stride=2,padding=0))
        
        # 每个decoder conv前放大一倍图像尺寸，缩小一倍通道数
        self.deconvs=nn.ModuleList()
        for i in range(len(channels)-2):
            self.deconvs.append(nn.ConvTranspose2d(channels[-i-1],channels[-i-2],kernel_size=2,stride=2))

        # 每个decoder conv block减少一倍通道数
        self.dec_convs=nn.ModuleList()
        for i in range(len(channels)-2):
            self.dec_convs.append(ConvBlock(channels[-i-1],channels[-i-2],time_emb_size,qsize,vsize,fsize,cls_emb_size))   # 残差结构

        # 还原通道数,尺寸不变
        self.output=nn.Conv2d(channels[1],img_channel,kernel_size=1,stride=1,padding=0)
        
    def forward(self,x,t,cls):  # cls是引导词（图片分类ID）
        # time做embedding
        t_emb=self.time_emb(t)

        # cls做embedding
        cls_emb=self.cls_emb(cls)
        
        # encoder阶段
        residual=[]
        for i,conv in enumerate(self.enc_convs):
            x=conv(x,t_emb,cls_emb)
            if i!=len(self.enc_convs)-1:
                residual.append(x)
                x=self.maxpools[i](x)
            
        # decoder阶段
        for i,deconv in enumerate(self.deconvs):
            x=deconv(x)
            residual_x=residual.pop(-1)
            x=self.dec_convs[i](torch.cat((residual_x,x),dim=1),t_emb,cls_emb)    # 残差用于纵深channel维
        return self.output(x) # 还原通道数
        
if __name__=='__main__':
    batch_x=torch.stack((train_dataset[0][0],train_dataset[1][0]),dim=0).to(DEVICE) # 2个图片拼batch, (2,1,48,48)
    batch_x=batch_x*2-1 # 像素值调整到[-1,1]之间,以便与高斯噪音值范围匹配
    batch_cls=torch.tensor([train_dataset[0][1],train_dataset[1][1]],dtype=torch.long).to(DEVICE)  # 引导ID

    batch_t=torch.randint(0,T,size=(batch_x.size(0),)).to(DEVICE)  # 每张图片随机生成diffusion步数
    batch_x_t,batch_noise_t=forward_diffusion(batch_x,batch_t)

    print('batch_x_t:',batch_x_t.size())
    print('batch_noise_t:',batch_noise_t.size())

    unet=UNet(img_channel=1).to(DEVICE)
    batch_predict_noise_t=unet(batch_x_t,batch_t,batch_cls)
    print('batch_predict_noise_t:',batch_predict_noise_t.size())