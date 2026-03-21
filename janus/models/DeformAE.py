
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import os
import numpy as np
import cv2
import glob

class DeformEncoder(nn.Module):
    def __init__(self):
        super(DeformEncoder, self).__init__()
        resnet18 = models.resnet18(weights=None)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,
        )
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.reshape_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.layer3 = resnet18.layer3
        self.reshape_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, deform):
        x = self.stem(deform)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.reshape_layer1(x)
        x = self.layer3(x)
        x = self.reshape_layer2(x)
        return x   # [B,128,H’,W’]


class DeformDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deformation_head = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=4),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5, stride=1, padding=2),
        )


class DeformAEInfer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DeformEncoder()
        self.decoder = DeformDecoder()

    def forward(self, deform):
        x = self.encoder(deform)
        deform_decoded = self.decoder.deformation_head(x)
        return deform_decoded


if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ckpt", "sharpa_wave_deform_encoder.pth")
    img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deform_imgs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeformAEInfer().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Checkpoint does not contain a valid state_dict")

    model_state = model.state_dict()
    load_state = {k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape}
    model.load_state_dict(load_state, strict=False)
    model.eval()

    # read all images in img_dir
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
    img_paths = [p for p in img_paths if os.path.isfile(p)]
    if not img_paths:
        raise FileNotFoundError(f"No images found in: {img_dir}")

    for img_path in img_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img_u8 = img.astype(np.uint8)
        img = img.astype(np.float32)
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device=device)
        print(f"Original deform max: {img.max()}, Original deform min: {img.min()}")
        with torch.no_grad():
            deform_decoded = model(img)
            deform_decoded = deform_decoded.detach().cpu().numpy()

        # visualize the deformation maps
        deform_decoded = np.clip(deform_decoded[0, 0], 0, 255.0)
        deform_decoded = deform_decoded.astype(np.uint8)
        print(f"Decoded deform max: {deform_decoded.max()}, Decoded deform min: {deform_decoded.min()}")

        vis = np.concatenate([img_u8, deform_decoded], axis=1)
        cv2.imshow("deform_input_vs_output", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
