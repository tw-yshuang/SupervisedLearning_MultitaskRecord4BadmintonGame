import torch
from torch import nn
from torchvision.models.efficientnet import efficientnet_v2_m, EfficientNet_V2_M_Weights, EfficientNet


def efficientnet_v2_m_forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    return x


EfficientNet.forward = efficientnet_v2_m_forward


class Conveter(nn.Module):
    def __init__(self, in_seq):
        super(Conveter, self).__init__()

        self.conv = nn.Sequential(
            # pw
            nn.Conv3d(in_seq, 3, kernel_size=(1, 1, 1), stride=1, padding=0, bias=False),
            # nn.LazyBatchNorm3d(),
            nn.BatchNorm3d(3),
            nn.ReLU(),
            nn.Conv3d(3, 1, kernel_size=(1, 1, 1), stride=1, padding=0, bias=False),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return torch.squeeze(x, dim=1)


class EffNet(nn.Module):
    def __init__(self, in_seq, output_classes, *args, **kwargs) -> None:
        super(EffNet, self).__init__(*args, **kwargs)

        self.converter = Conveter(in_seq)
        self.features = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        # self.features.forward = efficientnet_v2_m_forward
        self.linear = nn.Sequential(
            nn.Linear(self.features.lastconv_output_channels, output_classes),
        )

    def forward(self, x):
        x = self.converter(x)
        x = self.features(x)
        x = self.linear(x)
        return x
