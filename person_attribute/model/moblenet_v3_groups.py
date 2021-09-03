import torch
import torch.nn as nn
import torchvision

class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6(inplace)

    def forward(self, x):
        return x*self.relu6(x+3)/6

def ConvBNActivation(in_channels,out_channels,kernel_size,stride,activate):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=in_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True) if activate == 'relu' else HardSwish()
        )

def Conv1x1BNActivation(in_channels,out_channels,activate):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True) if activate == 'relu' else HardSwish()
        )

def Conv1x1BN(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

class SqueezeAndExcite(nn.Module):
    def __init__(self, in_channels, out_channels,se_kernel_size, divide=4):
        super(SqueezeAndExcite, self).__init__()
        mid_channels = in_channels // divide
        self.pool = nn.AvgPool2d(kernel_size=se_kernel_size,stride=1)
        self.SEblock = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=mid_channels),
            nn.ReLU6(inplace=True),
            nn.Linear(in_features=mid_channels, out_features=out_channels),
            HardSwish(inplace=True),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        out = self.pool(x)
        out = out.view(b, -1)
        out = self.SEblock(out)
        out = out.view(b, c, 1, 1)
        return out * x

class SEInvertedBottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride,activate, use_se, se_kernel_size=1):
        super(SEInvertedBottleneck, self).__init__()
        self.stride = stride
        self.use_se = use_se
        # mid_channels = (in_channels * expansion_factor)

        self.conv = Conv1x1BNActivation(in_channels, mid_channels,activate)
        self.depth_conv = ConvBNActivation(mid_channels, mid_channels, kernel_size,stride,activate)
        if self.use_se:
            self.SEblock = SqueezeAndExcite(mid_channels, mid_channels, se_kernel_size)

        self.point_conv = Conv1x1BNActivation(mid_channels, out_channels,activate)

        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)

    def forward(self, x):
        out = self.depth_conv(self.conv(x))
        if self.use_se:
            out = self.SEblock(out)
        out = self.point_conv(out)
        out = (out + self.shortcut(x)) if self.stride == 1 else out
        return out


class MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000,type='large'):
        super(MobileNetV3, self).__init__()
        self.type = type

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            HardSwish(inplace=True),
        )

        if type=='large':
            self.large_bottleneck = nn.Sequential(
                SEInvertedBottleneck(in_channels=16, mid_channels=16, out_channels=16, kernel_size=3, stride=1,activate='relu', use_se=False),
                SEInvertedBottleneck(in_channels=16, mid_channels=64, out_channels=24, kernel_size=3, stride=2, activate='relu', use_se=False),
                SEInvertedBottleneck(in_channels=24, mid_channels=72, out_channels=24, kernel_size=3, stride=1, activate='relu', use_se=False),
                SEInvertedBottleneck(in_channels=24, mid_channels=72, out_channels=40, kernel_size=5, stride=2,activate='relu', use_se=True, se_kernel_size=(28,16)),
                SEInvertedBottleneck(in_channels=40, mid_channels=120, out_channels=40, kernel_size=5, stride=1,activate='relu', use_se=True, se_kernel_size=(28,16)),
                SEInvertedBottleneck(in_channels=40, mid_channels=120, out_channels=40, kernel_size=5, stride=1,activate='relu', use_se=True, se_kernel_size=(28,16)),
                SEInvertedBottleneck(in_channels=40, mid_channels=240, out_channels=80, kernel_size=3, stride=1,activate='hswish', use_se=False),
                SEInvertedBottleneck(in_channels=80, mid_channels=200, out_channels=80, kernel_size=3, stride=1,activate='hswish', use_se=False),
                SEInvertedBottleneck(in_channels=80, mid_channels=184, out_channels=80, kernel_size=3, stride=2,activate='hswish', use_se=False),
                SEInvertedBottleneck(in_channels=80, mid_channels=184, out_channels=80, kernel_size=3, stride=1,activate='hswish', use_se=False),
                SEInvertedBottleneck(in_channels=80, mid_channels=480, out_channels=112, kernel_size=3, stride=1,activate='hswish', use_se=True, se_kernel_size=(14,8)),
                SEInvertedBottleneck(in_channels=112, mid_channels=672, out_channels=112, kernel_size=3, stride=1,activate='hswish', use_se=True, se_kernel_size=(14,8)),
                SEInvertedBottleneck(in_channels=112, mid_channels=672, out_channels=160, kernel_size=5, stride=2,activate='hswish', use_se=True,se_kernel_size=(7,4)),
                SEInvertedBottleneck(in_channels=160, mid_channels=960, out_channels=160, kernel_size=5, stride=1,activate='hswish', use_se=True,se_kernel_size=(7,4)),
                SEInvertedBottleneck(in_channels=160, mid_channels=960, out_channels=160, kernel_size=5, stride=1,activate='hswish', use_se=True,se_kernel_size=(7,4)),
            )

            self.large_last_stage = nn.Sequential(
                nn.Conv2d(in_channels=160, out_channels=960, kernel_size=1, stride=1),
                nn.BatchNorm2d(960),
                HardSwish(inplace=True),
                nn.AvgPool2d(kernel_size=(7,4), stride=1),
                # nn.Conv2d(in_channels=960, out_channels=1280, kernel_size=1, stride=1),
                # HardSwish(inplace=True),
            )
        else:
            self.small_bottleneck = nn.Sequential(
                SEInvertedBottleneck(in_channels=16, mid_channels=16, out_channels=16, kernel_size=3, stride=2,activate='relu', use_se=True, se_kernel_size=56),
                SEInvertedBottleneck(in_channels=16, mid_channels=72, out_channels=24, kernel_size=3, stride=2,activate='relu', use_se=False),
                SEInvertedBottleneck(in_channels=24, mid_channels=88, out_channels=24, kernel_size=3, stride=1,activate='relu', use_se=False),
                SEInvertedBottleneck(in_channels=24, mid_channels=96, out_channels=40, kernel_size=5, stride=2,activate='hswish', use_se=True, se_kernel_size=14),
                SEInvertedBottleneck(in_channels=40, mid_channels=240, out_channels=40, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=14),
                SEInvertedBottleneck(in_channels=40, mid_channels=240, out_channels=40, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=14),
                SEInvertedBottleneck(in_channels=40, mid_channels=120, out_channels=48, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=14),
                SEInvertedBottleneck(in_channels=48, mid_channels=144, out_channels=48, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=14),
                SEInvertedBottleneck(in_channels=48, mid_channels=288, out_channels=96, kernel_size=5, stride=2,activate='hswish', use_se=True, se_kernel_size=7),
                SEInvertedBottleneck(in_channels=96, mid_channels=576, out_channels=96, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=7),
                SEInvertedBottleneck(in_channels=96, mid_channels=576, out_channels=96, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=7),
            )
            self.small_last_stage = nn.Sequential(
                nn.Conv2d(in_channels=96, out_channels=576, kernel_size=1, stride=1),
                nn.BatchNorm2d(576),
                HardSwish(inplace=True),
                nn.AvgPool2d(kernel_size=7, stride=1),
                nn.Conv2d(in_channels=576, out_channels=1280, kernel_size=1, stride=1),
                HardSwish(inplace=True),
            )
        self.classify_iwbr1 = nn.Sequential(
            nn.Conv2d(in_channels=960, out_channels=96, kernel_size=1, stride=1),
            HardSwish(inplace=True),
        )
        self.classify_iwbr2 = nn.Sequential(
            nn.Conv2d(in_channels=960, out_channels=96, kernel_size=1, stride=1),
            HardSwish(inplace=True),
        )
        self.classify_iwbr3 = nn.Sequential(
            nn.Conv2d(in_channels=960, out_channels=96, kernel_size=1, stride=1),
            HardSwish(inplace=True),
        )
        self.classify_iwbr4 = nn.Sequential(
            nn.Conv2d(in_channels=960, out_channels=96, kernel_size=1, stride=1),
            HardSwish(inplace=True),
        )
        self.classify_iwbr5 = nn.Sequential(
            nn.Conv2d(in_channels=960, out_channels=96, kernel_size=1, stride=1),
            HardSwish(inplace=True),
        )
        self.classify_iwbr6 = nn.Sequential(
            nn.Conv2d(in_channels=960, out_channels=96, kernel_size=1, stride=1),
            HardSwish(inplace=True),
        )
        self.classify_iwbr7 = nn.Sequential(
            nn.Conv2d(in_channels=960, out_channels=96, kernel_size=1, stride=1),
            HardSwish(inplace=True),
        )
        self.classify_iwbr8 = nn.Sequential(
            nn.Conv2d(in_channels=960, out_channels=96, kernel_size=1, stride=1),
            HardSwish(inplace=True),
        )
        self.classify_iwbr9 = nn.Sequential(
            nn.Conv2d(in_channels=960, out_channels=96, kernel_size=1, stride=1),
            HardSwish(inplace=True),
        )
        self.classify_iwbr10 = nn.Sequential(
            nn.Conv2d(in_channels=960, out_channels=96, kernel_size=1, stride=1),
            HardSwish(inplace=True),
        )
        
        # self.classifier = nn.Conv2d(in_channels=1280, out_channels=10, kernel_size=1, stride=1)
        # self.lstm = nn.LSTM(input_size=1, hidden_size=17, num_layers=1,batch_first=True)
        self.linear1 = nn.Linear(96,10)
        self.linear2 = nn.Linear(96,10)
        self.linear3 = nn.Linear(96,10)
        self.linear4 = nn.Linear(96,10)
        self.linear5 = nn.Linear(96,10)
        self.linear6 = nn.Linear(96,10)
        self.linear7 = nn.Linear(96,10)
        self.linear8 = nn.Linear(96,10)
        self.linear9 = nn.Linear(96,10)
        self.linear10 = nn.Linear(96,10)
        self.reclassify_one1 = nn.Linear(10,2)
        self.reclassify_one2 = nn.Linear(10,2)
        self.reclassify_one3 = nn.Linear(10,2)
        self.reclassify_one4 = nn.Linear(10,2)
        self.reclassify_one5 = nn.Linear(10,2)
        self.reclassify_one6 = nn.Linear(10,2)

        self.reclassify_two9 = nn.Linear(10,2)
        self.reclassify_three7 = nn.Linear(10,3)
        self.reclassify_three8 = nn.Linear(10,3)
        self.reclassify_three10 = nn.Linear(10,3)
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)# /2
        if self.type == 'large':
            x = self.large_bottleneck(x)# /2^4
            x = self.large_last_stage(x)
        else:
            x = self.small_bottleneck(x)
            x = self.small_last_stage(x)

        #网络分为10个分支
        out_fmale       = self.linear1(self.classify_iwbr1(x).squeeze(-1).squeeze(-1))
        out_age         = self.linear2(self.classify_iwbr2(x).squeeze(-1).squeeze(-1))
        out_dirction    = self.linear3(self.classify_iwbr3(x).squeeze(-1).squeeze(-1))
        out_upclothing  = self.linear4(self.classify_iwbr4(x).squeeze(-1).squeeze(-1))
        out_downclothing= self.linear5(self.classify_iwbr5(x).squeeze(-1).squeeze(-1))
        out_hat         = self.linear6(self.classify_iwbr6(x).squeeze(-1).squeeze(-1))
        out_glass       = self.linear7(self.classify_iwbr7(x).squeeze(-1).squeeze(-1))
        out_handBag     = self.linear8(self.classify_iwbr8(x).squeeze(-1).squeeze(-1))
        out_shoulderbag = self.linear9(self.classify_iwbr9(x).squeeze(-1).squeeze(-1))
        out_carry       = self.linear10(self.classify_iwbr10(x).squeeze(-1).squeeze(-1))
        #每个分支分类数目可能不同
        pre_out_fmale = self.reclassify_one1(out_fmale)
        pre_out_age = self.reclassify_three7(out_age)
        pre_out_dirction = self.reclassify_three8(out_dirction)
        pre_out_upclothing = self.reclassify_two9(out_upclothing)
        pre_out_downclothing = self.reclassify_three10(out_downclothing)
        pre_out_hat = self.reclassify_one2(out_hat)
        pre_out_glass = self.reclassify_one3(out_glass)
        pre_out_handBag = self.reclassify_one4(out_handBag)
        pre_out_shoulderbag = self.reclassify_one5(out_shoulderbag)
        pre_out_carry = self.reclassify_one6(out_carry)
        
        # x,_ = self.lstm(out.squeeze(-1))
        # s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        # x = x.contiguous().view(s*b, h)
        # x = self.reclassify(x)
        # x = x.contiguous().view(s, b, -1)
        return out_fmale,out_hat,out_glass,out_handBag,out_shoulderbag,out_carry,out_age,out_dirction,out_upclothing,out_downclothing \
            ,pre_out_fmale,pre_out_hat,pre_out_glass,pre_out_handBag,pre_out_shoulderbag,pre_out_carry,pre_out_age,pre_out_dirction,pre_out_upclothing,pre_out_downclothing

if __name__ == '__main__':
    from torchvision import transforms
    from PIL import Image
    from merge_bn import *
    import numpy as np
    np.set_printoptions(suppress=True)

    mytransform = transforms.Compose([

        # transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        # transforms.RandomCrop(299),
        # transforms.Resize((299,299)),       #TODO:maybe need to change1
        transforms.ToTensor(),  # mmb,
    ]
    )
    # img = Image.open(r'D:\li_data\li_pa_100\release_data\release_data\000186.jpg')
    # img = mytransform(img)
    # # img = img.view(1, 3, 224, 224)
    # img = torch.unsqueeze(img, 0)
    # attri = ['Female', 'AgeOver60', 'Age18-60', 'AgeLess18', 'Front', 'Side', 'Back', 'Hat', 'Glasses','HandBag'
    #          'ShoulderBag', 'Backpack', 'HoldObjectsInFront', 'ShortSleeve', 'LongSleeve', 'UpperStride', 'UpperLogo',
    #          'UpperPlaid', 'UpperSplice', 'LowerStripe', 'LowerPattern', 'LongCoat', 'Trousers', 'Shorts', 'Skirt&Dress','boots'
    #          ]
    model = MobileNetV3(type='large', num_classes=26)
    model.load_state_dict(torch.load('../checkpoint/checkpoint_epoch_95'))
    model.eval()

    # model = torch.jit.trace(model, torch.randn(1,3,224,224))
    # model.save('best.pt')
    # fuse_module(model)
    # batch_size = 1  # 批处理大小
    # input_shape = (3, 224, 224)  # 输入数据
    # model.eval()
    #
    # x = torch.randn(batch_size, *input_shape)
    # export_onnx_file = "person_attri.onnx"
    # torch.onnx.export(model,
    #                   x,
    #                   export_onnx_file,
    #                   # opset_version=10,
    #                   verbose=True,
    #                   # do_constant_folding=True,	# 是否执行常量折叠优化
    #                   input_names=["input"],  # 输入名
    #                   output_names=["output"]  # 输出名
    #                   )
    # import onnx
    # from onnxsim import simplify
    #
    # # load your predefined ONNX model
    # model = onnx.load('person_attri.onnx')
    # model_simp, check = simplify(model)
    #
    # assert check, "Simplified ONNX model could not be validated"
    # onnx.save(model_simp, 'person_attri_simp.onnx')
    # a = torch.randn(2,3,224,224)
    # traced_script_module = torch.jit.trace(model, a)
    # traced_script_module.save('person.pt')
    #
    # # torch.jit.save(torch.jit.script(model, a), 'person.pt')
    #
    # img = Image.open(r'C:\Users\Glasssix-LQJ\Desktop\person_attri.jpg')
    # img = Image.open(r'C:\Users\Glasssix-LQJ\Desktop\person_attri.jpg')
    import cv2
    img = cv2.imread(r'C:\Users\Glasssix-LQJ\Desktop\person_attri.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.transpose(2, 0, 1)  # hwc转为chw
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    # img = mytransform(img)
    # img = img.view(1, 3, 224, 224)
    img = torch.unsqueeze(img, 0)
    # out = traced_script_module(img)
    out = model(img).detach().numpy()
    print(out)
