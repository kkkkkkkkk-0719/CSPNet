import torch
from torch import Tensor
from typing import List
from collections import OrderedDict
from torch import nn
import torch.utils.checkpoint as cp
import torch.nn.functional as F

class _Transition(nn.Sequential):
    def __init__(self,
                 num_input_features: int,
                 num_output_features: int):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_features,
                                          num_output_features,
                                          kernel_size=1,
                                          stride=1,
                                          bias=False))
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))

class _DenseLayer(nn.Module):
    """DenseBlock中的内部结构 DenseLayer: BN + ReLU + Conv(1x1) + BN + ReLU + Conv(3x3)"""
    def __init__(self,
                 num_input_features: int,
                 growth_rate: int,
                 bn_size: int,
                 drop_rate: float,
                 memory_efficient: bool = False):
        """
        :param input_c: 输入channel
        :param growth_rate: 论文中的 k = 32
        :param bn_size: 1x1卷积的filternum = bn_size * k  通常bn_size=4
        :param drop_rate: dropout 失活率
        :param memory_efficient: Memory-efficient版的densenet  默认是不使用的
        """
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(in_channels=num_input_features,
                                           out_channels=bn_size * growth_rate,
                                           kernel_size=1,
                                           stride=1,
                                           bias=False))
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size * growth_rate,
                                           growth_rate,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1,
                                           bias=False))
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        # 第一个DenseBlock inputs： 最后会生成 [16,32,56,56](输入) + [16,32,56,56]*5
        # concat_features=6个List的shape分别是: [16,32,56,56](输入)、[16,32,56,56]、[16,64,56,56]、[16,96,56,56]、[16,128,56,56]、[16,160,56,56]、[16,192,56,56]
        concat_features = torch.cat(inputs, 1)  # 该DenseBlock的每一个DenseLayer的输入都是这个DenseLayer之前所有DenseLayer的输出再concat
        # 之后的DenseBlock中的append会将每一个之前层输入加入inputs 但是这个concat并不是把所有的Dense Layer层直接concat到一起
        # 注意：这个concat和之后的DenseBlock中的concat非常重要，理解这两句就能理解DenseNet中密集连接的精髓

        bottleneck_output = self.conv1(self.relu1(self.norm1(concat_features))) # 一直是[16,128,56,56]
        return bottleneck_output

    @staticmethod
    def any_requires_grad(inputs: List[Tensor]) -> bool:
        """判断是否需要更新梯度（training）"""
        for tensor in inputs:
            if tensor.requires_grad:
                return True

        return False

    @torch.jit.unused
    def call_checkpoint_bottleneck(self, inputs: List[Tensor]) -> Tensor:
        """
        torch.utils.checkpoint: 用计算换内存（节省内存）。 详情可看： https://arxiv.org/abs/1707.06990
        torch.utils.checkpoint并不保存中间激活值，而是在反向传播时重新计算它们。 它可以应用于模型的任何部分。
        具体而言，在前向传递中,function将以torch.no_grad()的方式运行,即不存储中间激活值 相反,前向传递将保存输入元组和function参数。
        在反向传播时，检索保存的输入和function参数，然后再次对函数进行正向计算，现在跟踪中间激活值，然后使用这些激活值计算梯度。
        """
        def closure(*inp):
            return self.bn_function(inp)

        return cp.checkpoint(closure, *inputs)

    def forward(self, inputs: Tensor) -> Tensor:
        if isinstance(inputs, Tensor):  # 确保inputs的格式满足要求
            prev_features = [inputs]
        else:
            prev_features = inputs

        # 判断是否使用memory_efficient的densenet  and  是否需要更新梯度（training）
        # torch.utils.checkpoint不适用于torch.autograd.grad（）,而仅适用于torch.autograd.backward（）
        if self.memory_efficient and self.any_requires_grad(prev_features):
            # torch.jit 模式下不合适用memory_efficient
            if torch.jit.is_scripting():
                raise Exception("memory efficient not supported in JIT")

            # 调用efficient densenet  思路：用计算换显存
            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            # 调用普通的densenet  永远是[16,128,56,56]
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))  # 永远是[16,32,56,56]
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)

        return new_features

class _Csp_Transition(torch.nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Csp_Transition, self).__init__()
        self.add_module('norm', torch.nn.BatchNorm2d(num_input_features))
        self.add_module('relu', torch.nn.ReLU(inplace=True))
        self.add_module('conv', torch.nn.Conv2d(num_input_features, num_output_features,
                                                kernel_size=1, stride=1, bias=False))


class _Csp_DenseBlock(torch.nn.Module):
    def __init__(self,
                 num_layers,
                 num_input_features,
                 bn_size,
                 growth_rate,
                 drop_rate,
                 memory_efficient=False,
                 transition=False):
        """
        :param num_layers: 当前DenseBlock的Dense Layer的个数
        :param num_input_features: 该DenseBlock的输入Channel，开始会进行拆分，最后concat 每经过一个DenseBlock都会进行叠加
                                   叠加方式：num_features = num_features // 2 + num_layers * growth_rate // 2
        :param bn_size: 1x1卷积的filternum = bn_size*k  通常bn_size=4
        :param growth_rate: 指的是论文中的k  小点比较好  论文中是32
        :param drop_rate: dropout rate after each dense layer
        :param memory_efficient: If True, uses checkpointing. Much more memory efficient
        :param transition: 分支需不需Transition(csp transition)  stand/fusionlast=True  fusionfirst=False
        """
        super(_Csp_DenseBlock, self).__init__()

        self.csp_num_features1 = num_input_features // 2  # 平均分成两部分 第一部分直接传到后面concat
        self.csp_num_features2 = num_input_features - self.csp_num_features1  # 第二部分进行正常卷积等操作
        trans_in_features = num_layers * growth_rate

        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features=self.csp_num_features2 + i * growth_rate,  # 每生成一个DenseLayer channel增加growth_rate
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)
        self.transition = _Csp_Transition(trans_in_features, trans_in_features // 2) if transition else None

    def forward(self, x):
        # x = [B, C, H, W]
        # 拆分channel, 每次只用一半的channel（csp_num_features1）会继续进行卷积等操作  另一半（csp_num_features2）直接传到当前DenseBlock最后进行concat
        features = [x[:, self.csp_num_features1:, ...]]  # [16,32,56,56]（输入） [16,32,56,56]*6

        for name, layer in self.named_children():
            if 'denselayer' in name:  # 遍历所有denselayer层
                # new_feature: 永远是[16,32,56,56]
                new_feature = layer(features)
                features.append(new_feature)
        dense = torch.cat(features[1:], 1)  # 第0个是上一DenseBlock的输入，所以不用concat
        # 到这里分支DenseBlock结束

        if self.transition is not None:
            dense = self.transition(dense)  # 进行分支（csp transition）Transition

        return torch.cat([x[:, :self.csp_num_features1, ...], dense], 1)


class Csp_DenseNet(torch.nn.Module):
    def __init__(self,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 transitionBlock=True,
                 transitionDense=False,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=1000,
                 memory_efficient=False):
        """
        :param growth_rate: DenseNet论文中的k 通常k=32
        :param block_config: 每个DenseBlock中Dense Layer的个数  121=>(6, 12, 24, 16)
        :param num_init_features: 模型第一个卷积层（Dense Block之前的唯一一个卷积）Conv0 的channel  = 64
        :param transitionBlock: 分支需不需要Transition    transitionDense: 主路需不需要transition
               transitionBlock=True  +  transitionDense=True  =>  stand
               transitionBlock=False  +  transitionDense=True  =>  fusionfirst
               transitionBlock=True  +  transitionDense=False  =>  fusionlast
        :param bn_size: 1x1卷积的filternum = bn_size*k  通常bn_size=4
        :param drop_rate: dropout rate after each dense layer 默认为0 不用的
        :param num_classes: 数据集类别数
        :param memory_efficient: If True, uses checkpointing. Much more memory efficient  默认为False
        """
        super(Csp_DenseNet, self).__init__()

        self.growth_down_rate = 2 if transitionBlock else 1  # growth_down_rate这个变量好像没用到
        self.features = torch.nn.Sequential(OrderedDict([
            ('conv0', torch.nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', torch.nn.BatchNorm2d(num_init_features)),
            ('relu0', torch.nn.ReLU(inplace=True)),
            ('pool0', torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _Csp_DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                transition=transitionBlock
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            # 每执行了一个Dense Block就要对下一个Dense Block的输入进行更新（channel进行了叠加）


            # 这里num_features变换是代码的最核心的部分
            # num_features：每个DenseBlock的输出
            # 如果支路用了transition: num_features=(上一个DenseBlock输出//2 + num_layers * growth_rate) // 2
            #                       因为只要经过transition输出都会变为原来的一半
            # 如果支路没有用transition: num_features=上一个DenseBlock输出//2 + num_layers * growth_rate
            num_features = num_features // 2 + num_layers * growth_rate // 2 if transitionBlock\
                else num_features // 2 + num_layers * growth_rate


            # 主路需不需要transition(常见的DenseNet的那种transition)
            if (i != len(block_config) - 1) and transitionDense:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module('norm5', torch.nn.BatchNorm2d(num_features))
        self.classifier = torch.nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = torch.nn.functional.relu(features, inplace=True)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def _csp_densenet(growth_rate, block_config, num_init_features, model='fusionlast', **kwargs):
    """
    :param growth_rate: DenseNet论文中的k 通常k=32
    :param block_config: 每个DenseBlock中Dense Layer的个数  121=>(6, 12, 24, 16)
    :param num_init_features: 模型第一个卷积层（Dense Block之前的唯一一个卷积）Conv0 的channel
    :param model: 模型类型 有stand、fusionfirst、fusionlast三种
    :param **kwargs: 不定长参数  通常会传入 num_classes

    transitionBlock: 分支需不需要Transition    transitionDense: 主路需不需要transition
    transitionBlock=True  +  transitionDense=True  =>  stand
    transitionBlock=False  +  transitionDense=True  =>  fusionfirst
    transitionBlock=True  +  transitionDense=False  =>  fusionlast
    """
    if model == 'stand':
        return Csp_DenseNet(growth_rate, block_config, num_init_features,
                            transitionBlock=True, transitionDense=True, **kwargs)
    if model == 'fusionfirst':
        return Csp_DenseNet(growth_rate, block_config, num_init_features,
                            transitionBlock=False, transitionDense=True, **kwargs)
    if model == 'fusionlast':
        return Csp_DenseNet(growth_rate, block_config, num_init_features,
                            transitionBlock=True, transitionDense=False, **kwargs)
    raise ('please input right model keyword')


def csp_densenet121(growth_rate=32, block_config=(6, 12, 24, 16),
                    num_init_features=64, model='fusionlast', **kwargs):
    return _csp_densenet(growth_rate, block_config, num_init_features, model=model, **kwargs)


def csp_densenet161(growth_rate=48, block_config=(6, 12, 36, 24),
                    num_init_features=96, model='fusionlast', **kwargs):
    return _csp_densenet(growth_rate, block_config, num_init_features, model=model, **kwargs)


def csp_densenet169(growth_rate=32, block_config=(6, 12, 32, 32),
                    num_init_features=64, model='fusionlast', **kwargs):
    return _csp_densenet(growth_rate, block_config, num_init_features, model=model, **kwargs)

def csp_densenet201(growth_rate=32, block_config=(6, 12, 48, 32),
                    num_init_features=64, model='fusionlast', **kwargs):
    return _csp_densenet(growth_rate, block_config, num_init_features, model=model, **kwargs)

if __name__ == '__main__':
    """测试模型"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 可以输入变量model='stand/fusionfirst/fusionlast（默认）'自己选择三种模型
    model = csp_densenet121(num_classes=5, model='fusionlast')
    print(model)
