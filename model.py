import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from conformer.model import Conformer


########################################################################################
class SELayer2D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBasicBlock2D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer2D(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class CoarseFine(nn.Module):
    def __init__(self, in_channels, output_channel):
        super(CoarseFine, self).__init__()
        self.coarse1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=7, stride=1, bias=False, dilation=1, padding=3),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=1, bias=False, dilation=1, padding=2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.fine1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.merge_flow1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.coarse2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=7, stride=1, bias=False, padding=3, dilation=1),
            nn.GELU(),
            nn.Conv2d(64, output_channel, kernel_size=5, stride=1, bias=False, padding=2, dilation=1),
            nn.BatchNorm2d(output_channel),
            nn.GELU(),
        )
        self.fine2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(64, output_channel, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.GELU(),
        )
        self.merge_flow2 = nn.Sequential(
            nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )
        self.se_block = SEBasicBlock2D(output_channel, output_channel)

    def forward(self, mfcc):
        mfcc_coarse = self.coarse1(mfcc)
        mfcc_fine = self.fine1(mfcc)
        x_merge = mfcc_coarse + mfcc_fine
        x_merge = self.merge_flow1(x_merge)

        mfcc_coarse = self.coarse2(x_merge)
        mfcc_fine = self.fine2(x_merge)
        x_merge = mfcc_coarse + mfcc_fine + x_merge
        x_merge = self.merge_flow2(x_merge)

        feature = self.se_block(x_merge)
        return feature


class LinerDecoder(nn.Module):
    def __init__(self, input_dim, ouput_dim):
        super(LinerDecoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, 30 * 8)
        self.bn1 = nn.BatchNorm1d(30 * 8)
        self.drop_out = nn.Dropout(0.1)
        self.fc3 = nn.Linear(30 * 8, ouput_dim)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc3(self.drop_out(x))
        x = self.sigmod(x)
        return x


class CFSCNet(nn.Module):
    def __init__(self):
        super(CFSCNet, self).__init__()
        segment_length = 15
        output_chanel = 32
        in_channels = 3
        self.sample_rate = 4000

        self.mrmsffc = CoarseFine(in_channels, output_chanel)
        self.conformer = Conformer(num_classes=segment_length,
                                   input_dim=output_chanel,
                                   encoder_dim=output_chanel,
                                   num_encoder_layers=1)
        self.decoder = LinerDecoder(11552, segment_length)

        self.imu_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=64,
            melkwargs={
                "n_fft": int(self.sample_rate / 5),
                "n_mels": 64,
                "hop_length": int(self.sample_rate / 10),
                "mel_scale": "htk",
            },
        )

    def forward(self, imu):
        cnn_signal = self.imu_transform(imu)
        cnn_signal = self.mrmsffc(cnn_signal)
        rnn_signal = cnn_signal.reshape(cnn_signal.size(0), cnn_signal.size(1), -1)
        rnn_signal = rnn_signal.permute(0, 2, 1)
        rnn_signal = self.conformer(rnn_signal)
        rnn_signal = rnn_signal.reshape(rnn_signal.size(0), -1)
        ouput = self.decoder(rnn_signal)
        return ouput
