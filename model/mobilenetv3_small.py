import torch as t
from torch import nn
from torch.nn import functional as F

class Hswish(nn.Module):
    def __init__(self):
        super(Hswish, self).__init__()
    def forward(self, inp):
        x = inp*F.relu6((inp+3), inplace=True)/6
        return x

class Hard_sigmoid(nn.Module):
    def __init__(self):
        super(Hard_sigmoid, self).__init__()
    def forward(self, inp):
        return F.relu6((inp+3), inplace=True)/6


class SEBlock(nn.Module):
    def __init__(self, exp_size, reduction):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(exp_size, exp_size // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(exp_size // reduction, exp_size)
        self.h_sigmoid = Hard_sigmoid()

    def forward(self, inp):
        x = self.pool(inp)
        x = t.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.h_sigmoid(x)
        x = t.reshape(x, (x.shape[0], x.shape[1], 1, 1))
        return inp * x


class BottleNeck(nn.Module):
    def __init__(self, inchannels, outchannels, stride, exp_size, activation="RE", use_se=False, ksize=3):
        super(BottleNeck, self).__init__()
        self.stride = stride
        self.use_se = use_se
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.conv1 = self.__convBNNL(inchannels, exp_size, 1, 1, 1, activation)
        self.conv2 = self.__convBNNL(exp_size, exp_size, ksize, stride, exp_size, activation)
        self.se = SEBlock(exp_size, 4)
        self.conv3 = nn.Conv2d(exp_size, outchannels, 1, 1, padding=0)

    def __convBNNL(self, inchannels, outchannels, ksize, stride, groups, activation="RE"):
        padding = (ksize - 1) // 2
        if activation == 'RE':
            return nn.Sequential(nn.Conv2d(inchannels, outchannels, ksize, stride, padding, groups=groups),
                                 nn.BatchNorm2d(outchannels),
                                 nn.ReLU6(inplace=True))
        elif activation == 'HS':
            return nn.Sequential(nn.Conv2d(inchannels, outchannels, ksize, stride, padding, groups=groups),
                                 nn.BatchNorm2d(outchannels),
                                 Hswish())

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.conv2(x)
        if self.use_se:
            x = self.se(x)
        x = self.conv3(x)
        if self.stride == 1 and self.inchannels == self.outchannels:
            return inp + x
        else:
            return x


class MobileNetV3_small(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetV3_small, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1),
                                   nn.BatchNorm2d(16),
                                   Hswish())
        self.bneck1 = BottleNeck(16, 16, 2, 16, use_se=True)
        self.bneck2 = BottleNeck(16, 24, 2, 72)
        self.bneck3 = BottleNeck(24, 24, 1, 88)
        self.bneck4 = BottleNeck(24, 40, 2, 96, activation="HS", use_se=True, ksize=5)
        self.bneck5 = BottleNeck(40, 40, 1, 240, activation="HS", use_se=True, ksize=5)
        self.bneck6 = BottleNeck(40, 40, 1, 240, activation="HS", use_se=True, ksize=5)
        self.bneck7 = BottleNeck(40, 48, 1, 120, activation="HS", use_se=True, ksize=5)
        self.bneck8 = BottleNeck(48, 48, 1, 144, activation="HS", use_se=True, ksize=5)
        self.bneck9 = BottleNeck(48, 96, 2, 288, activation="HS", use_se=True, ksize=5)
        self.bneck10 = BottleNeck(96, 96, 1, 576, activation="HS", use_se=True, ksize=5)
        self.bneck11 = BottleNeck(96, 96, 1, 576, activation="HS", use_se=True, ksize=5)
        self.conv2 = nn.Sequential(nn.Conv2d(96, 576, 1, stride=1, padding=0),
                                   nn.BatchNorm2d(576),
                                   Hswish())
        self.pool = nn.AvgPool2d(7)
        self.conv3 = nn.Sequential(nn.Conv2d(576, 1024, 1, stride=1, padding=0),
                                   Hswish())
        self.conv4 = nn.Conv2d(1024, num_classes, 1)

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bneck1(x)
        x = self.bneck2(x)
        x = self.bneck3(x)
        x = self.bneck4(x)
        x = self.bneck5(x)
        x = self.bneck6(x)
        x = self.bneck7(x)
        x = self.bneck8(x)
        x = self.bneck9(x)
        x = self.bneck10(x)
        x = self.bneck11(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = t.reshape(x, (x.shape[0], x.shape[1]))
        return x


if __name__ == '__main__':
    # t.cuda.empty_cache()
    img = t.rand((1, 3, 224, 224), dtype=t.float)

    mobileNet = MobileNetV3_small(num_classes=2)
    # mobileNet = mobileNet.cuda()
    out = mobileNet(img)

    print(out.shape)
    # print(mobileNet)
