import torch.nn as nn
from models.FCN_backbone import FCNs_VGG


class FCNs_CPS(nn.Module):
    def __init__(self, in_ch, out_ch, backbone='vgg16_bn', pretrained=True):
        super(FCNs_CPS, self).__init__()
        self.name = "FCNs_CPS_" + backbone
        self.backbone = backbone
        self.branch1 = FCNs_VGG(in_ch, out_ch, backbone=backbone, pretrained=pretrained)
        self.branch2 = FCNs_VGG(in_ch, out_ch, backbone=backbone, pretrained=pretrained)

    def forward(self, data, step=1):
        if not self.training:
            pred1 = self.branch1(data)
            return pred1

        if step == 1:
            return self.branch1(data)
        elif step == 2:
            return self.branch2(data)

if __name__ == '__main__':
    model = FCNs_CPS(in_ch=3, out_ch=1)
    print(model)




