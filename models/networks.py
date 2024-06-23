import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision.models.vgg import vgg16, vgg19,vgg19_bn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, wide_resnet50_2, wide_resnet101_2
import torch.nn.functional as F
# from torchsummary import summary
from torchvision.models import mobilenet_v2
from models.efficientnet import model


# class EfficientNet(nn.Module):
#     def __init__(self):
#         super(EfficientNet, self).__init__()
#         efficient_net = model.EfficientNet.from_pretrained('efficientnet-b0')
#         self.efficient_net = efficient_net.eval()
#
#     def forward(self, input_):
#         with torch.no_grad():
#             features = self.efficient_net.extract_features(input_)
#         for feature in features:
#             print(feature.shape)
#         #
#         # f1_ = F.interpolate(out1, size=(64, 64), mode='bilinear', align_corners=True)
#         # f2_ = F.interpolate(out2, size=(64, 64), mode='bilinear', align_corners=True)
#         # f3_ = F.interpolate(out3, size=(64, 64), mode='bilinear', align_corners=True)
#         # f4_ = F.interpolate(out4, size=(64, 64), mode='bilinear', align_corners=True)
#         #
#         # f_ = torch.cat([f1_, f2_, f3_, f4_], dim=1)
#         return features
#
# model = EfficientNet()
# input_tensor = torch.rand(1,3,256,256)
# output_tensor =model(input_tensor)

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        mobilenet = mobilenet_v2(True)
        layers = mobilenet.features
        # for i in range(15):
        #     print(layers[i])
        #     print("=================================")
        self.layer1 = layers[:1]
        self.layer2 = layers[1:2]
        self.layer3 = layers[2:4]
        self.layer4 = layers[4:7]

    def forward(self, input_):
        out1 = self.layer1(input_)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        f1_ = F.interpolate(out1, size=(64, 64), mode='bilinear', align_corners=True)
        f2_ = F.interpolate(out2, size=(64, 64), mode='bilinear', align_corners=True)
        f3_ = F.interpolate(out3, size=(64, 64), mode='bilinear', align_corners=True)
        f4_ = F.interpolate(out4, size=(64, 64), mode='bilinear', align_corners=True)

        f_ = torch.cat([f1_, f2_, f3_, f4_], dim=1)
        return f_



class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg = vgg19(True)
        layers = vgg.features
        # print(layers)
        self.layer1 = layers[:5]
        self.layer2 = layers[5:10]
        self.layer3 = layers[10:19]
        self.layer4 = layers[19:28]

    def forward(self, input_):
        out1 = self.layer1(input_)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        f1_ = F.interpolate(out1, size=(64, 64), mode='bilinear', align_corners=True)
        f2_ = F.interpolate(out2, size=(64, 64), mode='bilinear', align_corners=True)
        f3_ = F.interpolate(out3, size=(64, 64), mode='bilinear', align_corners=True)
        f4_ = F.interpolate(out4, size=(64, 64), mode='bilinear', align_corners=True)

        f_ = torch.cat([f1_, f2_, f3_, f4_], dim=1)

        return f_

# class VGG(nn.Module):
#     def __init__(self):
#         super(VGG, self).__init__()
#         vgg = vgg19_bn(True)
#         layers = vgg.features
#         # print(layers)
#         self.layer1 = layers[:6]
#         self.layer2 = layers[6:13]
#         self.layer3 = layers[13:26]
#         self.layer4 = layers[26:39]
#         self.layer5 = layers[39:]
#
#     def forward(self, input_):
#         out1 = self.layer1(input_)
#         out2 = self.layer2(out1)
#         out3 = self.layer3(out2)
#         out4 = self.layer4(out3)
#         f1_ = F.interpolate(out1, size=(64, 64), mode='bilinear', align_corners=True)
#         f2_ = F.interpolate(out2, size=(64, 64), mode='bilinear', align_corners=True)
#         f3_ = F.interpolate(out3, size=(64, 64), mode='bilinear', align_corners=True)
#         f4_ = F.interpolate(out4, size=(64, 64), mode='bilinear', align_corners=True)
#
#         f_ = torch.cat([f1_, f2_, f3_, f4_], dim=1)
#         return f_

class Resnet34(nn.Module):
    def __init__(self):
        super(Resnet34, self).__init__()
        resnet = resnet34(True)

        modules = list(resnet.children())
        self.block1 = nn.Sequential(*modules[0:4])
        self.block2 = modules[4]
        self.block3 = modules[5]
        self.block4 = modules[6]


    def forward(self, input_):
        out1 = self.block1(input_)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        f1_ = F.interpolate(out1, size=(64, 64), mode='bilinear', align_corners=True)
        f2_ = F.interpolate(out2, size=(64, 64), mode='bilinear', align_corners=True)
        f3_ = F.interpolate(out3, size=(64, 64), mode='bilinear', align_corners=True)
        f4_ = F.interpolate(out4, size=(64, 64), mode='bilinear', align_corners=True)

        f_ = torch.cat([f1_,f2_, f3_, f4_], dim=1)
        return f_


class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        resnet = resnet50(True)

        modules = list(resnet.children())
        self.block1 = nn.Sequential(*modules[0:4])
        self.block2 = modules[4]
        self.block3 = modules[5]
        self.block4 = modules[6]


    def forward(self, input_):
        out1 = self.block1(input_)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)

        # f1_ = F.interpolate(out1, size=(64, 64), mode='bilinear', align_corners=True)
        # f2_ = F.interpolate(out2, size=(64, 64), mode='bilinear', align_corners=True)
        f3_ = F.interpolate(out3, size=(64, 64), mode='bilinear', align_corners=True)
        f4_ = F.interpolate(out4, size=(64, 64), mode='bilinear', align_corners=True)
   

        f_ = torch.cat([f3_, f4_], dim=1)
        return f_

class WideResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        wideresnet50 = wide_resnet50_2(True)
        modules = list(wideresnet50.children())
        self.block1 = nn.Sequential(*modules[0:4])
        self.block2 = modules[4]
        self.block3 = modules[5]
        self.block4 = modules[6]


    def forward(self, input_):
        out1 = self.block1(input_)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)

        # f1_ = F.interpolate(out1, size=(64, 64), mode='bilinear', align_corners=True)
        f2_ = F.interpolate(out2, size=(64, 64), mode='bilinear', align_corners=True)
        f3_ = F.interpolate(out3, size=(64, 64), mode='bilinear', align_corners=True)
        f4_ = F.interpolate(out4, size=(64, 64), mode='bilinear', align_corners=True)
        avg_fliter = torch.nn.AvgPool2d(3, 1, 1)
        f2_ = avg_fliter(f2_)
        f3_ = avg_fliter(f3_)
        f4_ = avg_fliter(f4_)
        f_ = torch.cat([f2_, f3_, f4_], dim=1)
        return f_

class Resnet101(nn.Module):
    def __init__(self):
        super(Resnet101, self).__init__()
        resnet = resnet101(True)

        modules = list(resnet.children())
        self.block1 = nn.Sequential(*modules[0:4])
        self.block2 = modules[4]
        self.block3 = modules[5]
        self.block4 = modules[6]


    def forward(self, input_):
        out1 = self.block1(input_)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)

        f1_ = F.interpolate(out1, size=(64, 64), mode='bilinear', align_corners=True)
        f2_ = F.interpolate(out2, size=(64, 64), mode='bilinear', align_corners=True)
        f3_ = F.interpolate(out3, size=(64, 64), mode='bilinear', align_corners=True)
        f4_ = F.interpolate(out4, size=(64, 64), mode='bilinear', align_corners=True)


        f_ = torch.cat([f1_, f2_, f3_, f4_], dim=1)
        f_ = F.normalize(f_)
        return f_


class WideResnet101(nn.Module):
    def __init__(self):
        super(WideResnet101, self).__init__()
        wideresnet101 = wide_resnet101_2(True)

        modules = list(wideresnet101.children())
        self.block1 = nn.Sequential(*modules[0:4])
        self.block2 = modules[4]
        self.block3 = modules[5]
        self.block4 = modules[6]

    def forward(self, input_):
        out1 = self.block1(input_)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)

        f1_ = F.interpolate(out1, size=(64, 64), mode='bilinear', align_corners=True)
        f2_ = F.interpolate(out2, size=(64, 64), mode='bilinear', align_corners=True)
        f3_ = F.interpolate(out3, size=(64, 64), mode='bilinear', align_corners=True)
        f4_ = F.interpolate(out4, size=(64, 64), mode='bilinear', align_corners=True)

        f_ = torch.cat([f1_, f2_, f3_, f4_], dim=1)
        f_ = F.normalize(f_)
        return f_

class D_VGG(nn.Module):
    def __init__(self):
        super(D_VGG, self).__init__()
        vgg = vgg19(True)
        layers = vgg.features
        # print(layers)
        self.layer1 = layers[:5]
        self.layer2 = layers[5:10]
        self.layer3 = layers[10:19]
        self.layer4 = layers[19:28]

    def forward(self, input_):
        out1 = self.layer1(input_)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        f3_ = F.interpolate(out3, size=(64, 64), mode='bilinear', align_corners=True)
        f4_ = F.interpolate(out4, size=(64, 64), mode='bilinear', align_corners=True)

        f_ = torch.cat([f3_, f4_], dim=1)

        return f_

class IMAGE(nn.Module):
    def __init__(self):
        super(IMAGE, self).__init__()

    def forward(self, input_):
        input_ = F.interpolate(input_, size=(256, 256), mode='bilinear', align_corners=True)

        return input_

if __name__ == '__main__':
    a = torch.rand((1, 3, 256, 256))
    pre_fea = D_VGG()
    print(pre_fea(a).shape)