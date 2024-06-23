import torch
import torch.nn as nn
import torchvision

class Doubleconv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = Doubleconv(in_channels=1792, out_channels=512)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.layer2 = Doubleconv(in_channels=512, out_channels=256)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.layer3 = Doubleconv(in_channels=256, out_channels=128)
        self.layer4 = Doubleconv(in_channels=128, out_channels=64)

        self.ouput_layer = nn.Sequential(nn.Conv2d(64, 3, kernel_size=3, padding=1, bias=False))



    def forward(self, input_):
        out1 = self.layer1(input_)
        out1 = self.up1(out1)

        out2 = self.layer2(out1)
        out2 = self.up2(out2)

        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out5 = self.ouput_layer(out4)


        return out5

if __name__ =="__main__":
    image = torch.rand(1,1792,64,64)
    model = Decoder()
    output = model(image)
    print(output.shape)