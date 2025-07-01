## 导入对应的包
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.data
import torch

## 这个是卷积模块
class conv_block(nn.Module):
    """Conv module
    Args:
        in_ch : int
        out_ch : int
    """
    
    def __init__(self,in_ch : int , out_ch : int)->None:
        super(conv_block , self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels= in_ch ,
                      out_channels= out_ch,
                      kernel_size=  3 ,
                      stride = 1,
                      padding= 1,
                      bias= True
                      ),
            nn.BatchNorm2d(num_features= out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=  out_ch,
                      out_channels= out_ch,
                      kernel_size= 3,
                      stride= 1,
                      padding= 1,
                      bias=True
                      ),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x : torch.Tensor)->torch.Tensor:
        x = self.conv(x)
        return x
    
    
class up_conv(nn.Module):
    def __init__(self, in_ch : int , out_ch : int)->None:
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2 , mode="bilinear"),
            nn.Conv2d(in_channels= in_ch,
                      out_channels= out_ch,
                      kernel_size= 3,
                      stride= 1,
                      padding = 1,
                      bias= True
                      ),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x  = self.up(x)
        return x



