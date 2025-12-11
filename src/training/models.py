import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


# ---------------------------------------------------------
# DeepLabV3 (ResNet-50 backbone)
# ---------------------------------------------------------
def get_deeplab(num_classes=19):
    model = deeplabv3_resnet50(weights="DEFAULT")
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


# ---------------------------------------------------------
# Fast-SCNN (lightweight real-time segmentation) simplified implementation 
# ---------------------------------------------------------
class FastSCNN(nn.Module):
    # Extremely compact real-time semantic segmentation model.
    # This is a standard implementation used in tutorials/projects.
    def __init__(self, num_classes=19):
        super().__init__()

        # Learning downsample block
        self.down = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 48, 3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),

            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Global feature extractor (simplified)
        self.global_feat = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Classifier
        self.classifier = nn.Conv2d(128, num_classes, 1)

        # Simple upsampling
        self.up = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)

    def forward(self, x):
        x = self.down(x)
        x = self.global_feat(x)
        x = self.classifier(x)
        x = self.up(x)
        return {"out": x}


def get_fastscnn(num_classes=19):
    return FastSCNN(num_classes=num_classes)


# ---------------------------------------------------------
# BiSeNetV2 (lightweight + good accuracy) simplified implementation
# ---------------------------------------------------------
class BiSeNetV2(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()

        # Detail branch (high-resolution)
        self.detail = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Semantic branch (context)
        self.semantic = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # Fusion
        self.fusion = nn.Conv2d(128 + 64, 256, 1)

        # Classifier
        self.classifier = nn.Conv2d(256, num_classes, 1)

        self.up = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)

    def forward(self, x):
        d = self.detail(x)
        s = self.semantic(x)

        # Resize semantic to match detail
        s_up = torch.nn.functional.interpolate(s, size=d.shape[2:], mode="bilinear")

        fused = torch.cat([d, s_up], dim=1)
        fused = self.fusion(fused)

        out = self.classifier(fused)
        out = self.up(out)
        return {"out": out}


def get_bisenetv2(num_classes=19):
    return BiSeNetV2(num_classes=num_classes)

# ---------------------------------------------------------
# Baseline CNN (simple custom segmentation model)
# ---------------------------------------------------------
class BaselineCNN(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2)  # downsample by 2
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, num_classes, 1)  # produce logits
        )

        # Upsample to restore original size
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.up(x)
        return {"out": x}


def get_baselinecnn(num_classes=19):
    return BaselineCNN(num_classes=num_classes)

# -----------------------------
# MobileNetV3-LRASPP imported full
# -----------------------------
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

def get_mobilenet(num_classes=19):
    """Official lightweight segmentation model from TorchVision."""
    model = lraspp_mobilenet_v3_large(weights="DEFAULT")

    # Replace classifier layers to match our number of classes
    model.classifier.low_classifier = nn.Conv2d(40, num_classes, 1)
    model.classifier.high_classifier = nn.Conv2d(128, num_classes, 1)

    return model

# -----------------------------
# 2. ICNet (via MMseg)
# -----------------------------
# from mmseg.apis import init_model

# def get_icnet(num_classes=19, device="cpu"):
#     """
#     Load ICNet from MMsegmentation with pretrained weights.
#     Wrapped so it returns {'out': tensor}.
#     """
#     # Pre-trained config from MMsegmentation model zoo
#     config = "icnet_r18-d8_832x832_80k_cityscapes.py"
#     checkpoint = "icnet_r18-d8_832x832_cityscapes_20210925_094422.pth"

#     # Initialize model
#     mmseg_model = init_model(config, checkpoint, device=device)

#     # Wrap model so forward() returns {"out": pred_tensor}
#     class ICNetWrapper(nn.Module):
#         def __init__(self, model):
#             super().__init__()
#             self.model = model

#         def forward(self, x):
#             preds = self.model.encode_decode(
#                 x,
#                 img_metas=[{
#                     "ori_shape": x.shape[2:], 
#                     "img_shape": x.shape[2:], 
#                     "pad_shape": x.shape[2:],
#                     "scale_factor": 1.0,
#                     "flip": False
#                 }]
#             )   
#             # preds: (N, H, W)
#             preds = torch.from_numpy(preds).long().to(x.device)
#             return {"out": preds}

#     return ICNetWrapper(mmseg_model)
