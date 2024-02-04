
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base
from basicsr.models.archs.dynamic_region_conv import DRConv2d
from einops import rearrange
from pdb import set_trace as stx
import numbers


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
class SG_MSA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(SG_MSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x, seg_feats):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        seg_feats = rearrange(seg_feats, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = v * seg_feats

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class SGTB(nn.Module):
    def __init__(self, dim, num_blocks, num_heads=4, ffn_expansion_factor= 2.66,
                      bias= False, LayerNorm_type= 'BiasFree'):
        super(SGTB, self).__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([LayerNorm(dim, LayerNorm_type), LayerNorm(dim, LayerNorm_type),
            SG_MSA(dim, num_heads, bias),
            LayerNorm(dim, LayerNorm_type),
            FeedForward(dim, ffn_expansion_factor, bias)]))

    def forward(self, input_list):
        x, seg_feats = input_list[0],input_list[1]
        
        for (norm1, norm2, attn, ffn, norm3) in self.blocks:
            x = x + attn(norm1(x), norm2(seg_feats))
            x = x + ffn(norm3(x))

        return x
    
##########################################################################
class SegmentationGuidedEnhanceNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.seg_feats_downs = nn.ModuleList()

        self.level_count = len(enc_blk_nums)

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                
                    SGTB(chan,num)
                
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )

            self.seg_feats_downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = SGTB(dim=chan, num_blocks=middle_blk_num)
            

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                SGTB(chan, num)
                
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, seg_feats):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)


        encs = []
        seg_feats_store_list = []

        for encoder, down, seg_feats_down in zip(self.encoders, self.downs, self.seg_feats_downs):
            x = encoder([x, seg_feats])
            seg_feats_store_list.append(seg_feats)
            encs.append(x)
            x = down(x)
            seg_feats = seg_feats_down(seg_feats)

        x = self.middle_blks([x,seg_feats])

        level_curr = 0
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            seg_feats = seg_feats_store_list[self.level_count-1-level_curr]
            x = decoder([x, seg_feats])
            level_curr = level_curr + 1

        x = self.ending(x)
        x = x + inp[:,:3,:,:]

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

##########################################################################
class EdgeAwareFeatureExtractor(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3): 
        super(EdgeAwareFeatureExtractor, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)
        
        self.dr_conv = DRConv2d(n_fea_middle, n_fea_middle, kernel_size=1)

    def forward(self, rgb, edge):

        input = torch.cat([rgb,edge], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.dr_conv(x_1)
        return illu_fea

##########################################################################
class SGF(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.feature_converter = nn.Conv2d(in_channels=1, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        
        self.estimator = EdgeAwareFeatureExtractor(width)
        
        self.enhancer = SegmentationGuidedEnhanceNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num, enc_blk_nums=enc_blk_nums, dec_blk_nums=enc_blk_nums)

    def forward(self, inp):
        B, C, H, W = inp.shape

        img_rgb = inp[:,0:3,:,:]
        seg_map = inp[:,3:4,:,:]

        seg_feats = self.estimator(img_rgb, seg_map)
        
        x = self.enhancer(inp, seg_feats)

        return x

##########################################################################
if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis
    img_channel = 4
    width = 32

    enc_blks = [2, 2, 2, 2]
    middle_blk_num = 4
    dec_blks = [2, 2, 2, 2]
    
    net = SGF(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks).cuda()

    inp_img = torch.randn(1, 4, 256, 256).cuda()
    output = net(inp_img)
    print(output.shape)
    flops = FlopCountAnalysis(net,inp_img)
    n_param = sum([p.nelement() for p in net.parameters()])
    print(f'GMac:{flops.total()/(1024*1024*1024)}')
    print(f'Params:{n_param}')

