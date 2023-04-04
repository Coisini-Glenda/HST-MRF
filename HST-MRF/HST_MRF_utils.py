import torch.nn as nn
import torch.nn.functional as F
from data_utils import *
from einops import rearrange
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from swintrans import SwinTransformerBlock as HST

'''
HST-MRF
'''

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class M_conv(nn.Module):
    def __init__(self, channels_in, channels_out, args):
        super().__init__()
        self.args = args
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels_in, channels_in, (2, 2), stride=2, padding=0),
            DoubleConv(channels_in, channels_out, k=3, p=1, d=1),
            # nn.GroupNorm(num_channels=channels_out, num_groups=args.num_groups),
            # nn.Conv2d(channels_out, channels_out, (2, 2), stride=2, padding=0)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels_in, channels_in, (2, 2), stride=2, padding=0),
            DoubleConv(channels_in, channels_out, k=3, p=2, d=2),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv1[0].weight)
        nn.init.kaiming_normal_(self.conv2[0].weight)
        # nn.init.kaiming_normal_(self.conv3[0].weight)

    def forward(self, img1, img2):
        x1 = self.conv1(img1)
        x2 = self.conv2(img2)
        return x1, x2


class patch_embedding(nn.Module):
    '''
    基于softmax的注意力加权通道flatten
    输入维度为：[B, C, H, W] 例如这里: C=10，H=W=256，输入维度为：[B, C, 256, 256]
    输出维度为：[B, (H/patch_size * W/patch_size), C] 例如这里：patch_size=2，那么输出维度为：[B, (32*32), C]
    '''

    def __init__(self, args):
        super().__init__()
        self.patch_size = args.patch_size

    def forward(self, x):
        '''
        x: (B, C, H, W)
        经过切块后，维度转换为： (B, (H/patch_size * W/patch_size), C, (patch_size, patch_size))
        其中 (H/patch_size * W/patch_size) 表示块的数量，也是特征图的分辨率；
        (patch_size, patch_size) 表示每个块的分辨率。
        '''
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) c (p1 p2)', p1=self.patch_size, p2=self.patch_size)
        patch_weight = F.softmax(x, dim=-1)
        # nn.Softmax(dim=-1)(x)  # 维度为： [b (h*w) c (p1*p2)]
        x = x * patch_weight  # 维度为： [b (h*w) c]
        x = x.sum(dim=-1)
        return x


class Parallel(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        return sum([fn(x) for fn in self.fns])

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, args):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 2*dim),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(2 * dim, dim),
            nn.Dropout(args.dropout),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.net[0].weight)
        nn.init.xavier_normal_(self.net[3].weight)
        nn.init.normal_(self.net[0].bias, std=1e-6)
        nn.init.normal_(self.net[3].bias, std=1e-6)

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, args):
        super().__init__()
        inner_dim = args.head_dim * args.num_head
        project_out = not (args.num_head == 1 and args.head_dim == dim)

        self.heads = args.num_head
        self.scale = (2*args.head_dim) ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(args.dropout)

        self.to_qkv1 = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_qkv2 = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(args.dropout)
        ) if project_out else nn.Identity()

    def forward(self, x1, x2):
        qkv1 = self.to_qkv1(x1).chunk(3, dim=-1)
        qkv2 = self.to_qkv2(x2).chunk(3, dim=-1)

        q1, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv1)
        q2, k2, v2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv2)

        dots = self.scale*(torch.matmul(q1, k1.transpose(-1, -2))+torch.matmul(q2, k2.transpose(-1, -2)))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out1 = torch.matmul(attn, v1)
        out2 = torch.matmul(attn, v2)

        out1 = rearrange(out1, 'b h n d -> b n (h d)')
        out2 = rearrange(out2, 'b h n d -> b n (h d)')
        return self.to_out(torch.cat((out1, out2), dim=1))

class Transformer_block(nn.Module):
    def __init__(self, dim, args):
        super().__init__()

        self.Norm = nn.LayerNorm(dim)
        self.attn_block = Attention(dim, args)
        self.ff_block = FeedForward(dim, args)

    def forward(self, x1, x2):
        x1 = self.Norm(x1)
        x2 = self.Norm(x2)
        x = self.attn_block(x1, x2) + torch.cat((x1,x2), dim=1)
        x = self.Norm(x)
        x = self.ff_block(x) + x
        return x

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature. 输入特征的分辨率
        channels (int): Number of input channels. 输入的通道数
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, channels, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.channels = channels
        self.reduction = nn.Linear(4 * channels, 2 * channels, bias=False)
        self.norm = norm_layer(4 * channels)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x  # 输出维度为：x: B H/2*W/2 2*C

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, d=1, k=3, p=1, ng=32):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=k, padding=p, bias=False, dilation=d),
            nn.GroupNorm(num_channels=mid_channels, num_groups=ng),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=k, padding=p, bias=False, dilation=d),
            nn.GroupNorm(num_channels=mid_channels, num_groups=ng),
            nn.ReLU(inplace=True)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.double_conv[0].weight)
        nn.init.kaiming_normal_(self.double_conv[3].weight)

    def forward(self, x):
        return self.double_conv(x)

class Fusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv = DoubleConv(in_channels, in_channels//2)

    def forward(self, x1, x2, x3, x4=None):
        '''
        输入维度为: B C H W
        :return:
        '''
        x1_exp = torch.exp(x1)
        x2_exp = torch.exp(x2)
        x3_exp = torch.exp(x3)
        x1_weight = x1_exp / (x1_exp + x2_exp+x3_exp)
        x2_weight = x2_exp / (x1_exp + x2_exp+x3_exp)
        x3_weight = x3_exp/(x1_exp + x2_exp+x3_exp)
        x1 = torch.mul(x1, x1_weight)
        x2 = torch.mul(x2, x2_weight)
        x3 = torch.mul(x3, x3_weight)

        if x4==None:
            return x1+x2+x3
        else:
            x = self.conv(torch.cat(((x1+x2+x3), x4), dim=1))
            return x


class MBP(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = DoubleConv(in_channels, in_channels)
        self.conv2 = DoubleConv(2*in_channels, in_channels)

    def forward(self, x1, x2, x4=None):
        x = x1 * x2
        x = self.conv1(x)
        if x4==None:
            return x
        else:
            x = self.conv2(torch.cat((x, x4), dim=1))
            return x

class MLP(nn.Module):
    def __init__(self, dim, args):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(dim//2, dim),
            nn.Dropout(args.dropout),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.net[0].weight)
        nn.init.xavier_normal_(self.net[3].weight)
        nn.init.normal_(self.net[0].bias, std=1e-6)
        nn.init.normal_(self.net[3].bias, std=1e-6)

    def forward(self, x):
        return self.net(x)

class Soft_channel_attention(nn.Module):
    def __init__(self, dim, args):
        super().__init__()
        self.MLP = MLP(dim, args)

    def forward(self, x):
        '''
        :param x: 经过融合后的图像，维度为 B C H W
        :return:
        '''
        x1 = rearrange(x, 'b c h w -> b c (h w)')
        x_weight = F.softmax(x1, dim=-1)
        x_weight = rearrange(x_weight, 'b c (h w) -> b c h w', h=int(x_weight.size()[-1]**0.5))
        x1 = torch.sum(torch.mul(x, x_weight), dim=(2,3), keepdim=False)
        x1 = self.MLP(x1)
        x1 = rearrange(x1, 'b c -> b c 1 1')
        x1 = x + torch.mul(x, x1)
        return x1

class model(nn.Module):
    def __init__(self, args):
        super(model, self).__init__()
        channels = args.channels
        H = args.img_size

        # 第 1 阶段
        self.conv11 = DoubleConv(3, channels, k=3, p=1, d=1)
        self.conv12 = DoubleConv(3, channels,k=3, p=2, d=2)
        # self.conv13 = DoubleConv(3, channels, k=3, p=3, d=3)

        # 第 2 阶段
        self.conv2 = M_conv(channels, 2*channels, args)

        # 第 3 阶段
        self.patch_embedding = patch_embedding(args)   # 输入：(B 2C H/4 W/4)，输出：B (H/8 * W/8) 2C
        self.linear31 = nn.Linear(2*channels, 4*channels, bias=True)
        self.linear32 = nn.Linear(2*channels, 4*channels, bias=True)

        self.trans3 = HST(dim=4*channels, input_resolution=(H//4, H//4))#量级ST

        # 第 4 阶段
        self.patch_merge41 = PatchMerging((H//4, H//4), 4*channels)   # 输入：(H/8, W/8, 2C)，输出：(H/16, W/16, 4C)
        self.patch_merge42 = PatchMerging((H//4, H//4), 4*channels)

        self.trans4 = HST(dim=8*channels, input_resolution=(H//8,H//8))#量级ST

        # 第 5 阶段
        self.patch_merge51 = PatchMerging((H//8, H//8), 8 * channels)  # 输入：(H/16, W/16, 4C)，输出：(H/32, W/32, 8C)
        self.patch_merge52 = PatchMerging((H//8, H//8), 8 * channels)

        self.trans5 = HST(dim=16*channels, input_resolution=(H//16, H//16))#量级ST

        # 解码器与编码器对应的第 5 阶段
        self.fusion5 = MBP(16*channels)

        self.ca5 = Soft_channel_attention(16*channels, args)
        self.up5 = nn.ConvTranspose2d(16*channels, 16*channels // 2, kernel_size=2, stride=2)

        # 解码器与编码器对应的第 4 阶段
        self.fusion4 = MBP(8*channels)
        self.ca4 = Soft_channel_attention(8*channels, args)
        self.up4 = nn.ConvTranspose2d(8*channels, 8*channels//2, kernel_size=2, stride=2)

        # 解码器与编码器对应的第 3 阶段
        self.fusion3 = MBP(4 * channels)
        self.ca3 = Soft_channel_attention(4*channels, args)
        self.up3 = nn.ConvTranspose2d(4 * channels, 4 * channels // 2, kernel_size=2, stride=2)

        # 解码器与编码器对应的第 2 阶段

        self.fusion2 = MBP(2 * channels)

        self.ca2 = Soft_channel_attention(2*channels, args)
        self.up2 = nn.ConvTranspose2d(2 * channels, 2 * channels // 2, kernel_size=2, stride=2)

        # 解码器与编码器对应的第 1 阶段

        self.fusion1 = MBP(channels)

        self.ca1 = Soft_channel_attention(channels, args)
        self.conv = DoubleConv(channels, 1, ng=1)

        self.loss1 = nn.Sequential(
            nn.Conv2d(channels * 16, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Upsample(scale_factor=16)
        )
        self.loss2 = nn.Sequential(
            nn.Conv2d(channels * 2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, img):
        '''
        img 输入维度为： B 3 H W
        '''
        # 解码器阶段
        ## 第 1 阶段
        x11 = self.conv11(img)
        x12 = self.conv12(img)
        # x13 = self.conv13(img)
        del img
        ## 第 2 阶段
        x21,x22= self.conv2(x11,x12)   # B 2C H/2 W/2

        ## 第 3 阶段
        x31 = self.patch_embedding(x21)    # B (H/4 * W/4) 2C
        x32 = self.patch_embedding(x22)

        x31 = self.linear31(x31)  # B (H/4 * W/4) 4C
        x32 = self.linear32(x32)


        x31, x32 = self.trans3(x31, x32)

        ## 第 4 阶段
        x31_reverse = self.reverse_patch(x31)
        x32_reverse = self.reverse_patch(x32)

        x41 = self.patch_merge41(x31)
        x42 = self.patch_merge42(x32)

        x41, x42 = self.trans4(x41, x42)

        del x31, x32
        ## 第 5 阶段
        x41_reverse = self.reverse_patch(x41)
        x42_reverse = self.reverse_patch(x42)

        x51 = self.patch_merge51(x41)
        x52 = self.patch_merge52(x42)

        x51, x52 = self.trans5(x51, x52)

        # 解码器
        ## 对应的第 5 阶段
        x51_reverse = self.reverse_patch(x51)
        x52_reverse = self.reverse_patch(x52)

        del x41, x42,x51, x52
        x_fusion = self.fusion5(x51_reverse, x52_reverse)      # B 16C H/16 W/16

        l5 = self.loss1(x_fusion)
        x_fusion = self.up5(self.ca5(x_fusion))    # B 8C H/8 W/8


        del x51_reverse, x52_reverse
        ## 对应的第 4 阶段
        x_fusion = self.fusion4(x41_reverse, x42_reverse,x_fusion)  # B 8C H/8 W/8
        x_fusion = self.up4(self.ca4(x_fusion))   # B 4C H/4 W/4

        del x41_reverse, x42_reverse
        ## 对应的第 3 阶段
        x_fusion = self.fusion3(x31_reverse, x32_reverse, x_fusion)   # B 4C H/4 W/4

        x_fusion = self.up3(self.ca3(x_fusion))     # B 2C H/2 W/2

        l3 = self.loss2(x_fusion)
        del x31_reverse, x32_reverse
        ## 对应的第 2 阶段
        x_fusion = self.fusion2(x21, x22,x_fusion)   # B 2C H/2 W/2
        x_fusion = self.up2(self.ca2(x_fusion))  # B C H w

        del x21, x22
        ## 对应的第 1 阶段
        x_fusion = self.fusion1(x11,x12,x_fusion)   # B C H/2 W/2
        pred = self.conv(self.ca1(x_fusion))

        del x_fusion, x11, x12

        return pred, l5, l3

    def reverse_patch(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h=int(x.size()[1] ** 0.5))


































