import torch
import torch.nn as nn
# from utils.modules import DummyModule
import torch.nn as nn

class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()

    def forward(self, x):
        return x

def fuse(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    beta = bn.weight
    gamma = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * beta + gamma

    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=True,
        padding_mode=conv.padding_mode
    )
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv


def fuse_module(m):
    children = list(m.named_children())
    # print(children)
    conv = None
    conv_name = None

    for name, child in children:
        # print(child)
        if isinstance(child, nn.BatchNorm2d) and conv:
            bc = fuse(conv, child)
            m._modules[conv_name] = bc
            m._modules[name] = DummyModule()
            conv = None
        elif isinstance(child, nn.Conv2d):
            conv = child
            conv_name = name
        else:
            fuse_module(child)


def validate(net, input_, cuda=True):
    net.eval()
    # if cuda:
    #     input_ = input_.cuda()
    #     net.cuda()
    import time
    s = time.time()
    a,b,c,d = net(input_)
    # if cuda:
    #     torch.cuda.synchronize()
    print(time.time() - s)
    fuse_module(net)
    # print(mbnet)
    s = time.time()
    e, f, g, h= net(input_)
    # if cuda:
    #     torch.cuda.synchronize()
    print(time.time() - s)
    return (b - f).abs().max().item()


if __name__ == '__main__':
    import torchvision
    from model.model_pfld import PFLDInference1
    # net = torchvision.models.mobilenet_v2(pretrained=True)
    net = PFLDInference1()
    net.load_state_dict(torch.load('../param/model_pfld_new_param8.pth'))
    # net.eval()
    # # mbnet = torchvision.models.mobilenet_v2(True)
    # mbnet = PFLDInference1()
    # mbnet.load_state_dict(torch.load('../param/FP_model_8_param14.pth'))

    # mbnet.load_state_dict(torch.load('../utils/merge_all.pth'))
    # mbnet.eval()
    # print(list(mbnet.modules()))
    # for m in mbnet.modules():
    #     if type(m) == FAN1:
    #         print(m)

    # mbnet.load_state_dict(torch.load('D:/li/head_angel/param/merge_all_new990.pth'))
    # mbnet.load_state_dict(torch.load('../param/model_8_param14.pth'))
    # torch.save(mbnet, 'first.pth')
    net.eval()
    # net.eval()

    # fuse_module(net)
    # torch.save(net, 'mobile_2.pt')
    fuse_module(net)
    torch.jit.save(torch.jit.script(net), 'FPLD.pt')
    # print(validate(mbnet, torch.randn(1, 3, 64, 64), True))
    # a = torch.randn(1, 3, 64, 64)
    # net = torch.load('all.pt')
    # net.eval()
    # m = net(a)
    # print(m)



















