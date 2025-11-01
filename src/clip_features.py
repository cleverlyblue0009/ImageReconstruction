import torch
import torchvision.transforms as T
from torchvision.models import resnet50
import numpy as np
import cv2


class FrameEncoder:
    """
    Extracts deep features for each frame using a pretrained ResNet50 backbone.
    Outputs a 2048-D normalized embedding per frame.
    """

    def __init__(self, device=None, batch=32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch = batch

        # Load pretrained model
        model = resnet50(weights="IMAGENET1K_V2")
        model.fc = torch.nn.Identity()
        model.eval().to(self.device)
        self.model = model

        # Transform pipeline
        self.tf = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def encode(self, frames):
        """
        Takes a list of frames (BGR NumPy arrays) → returns Nx2048 normalized feature matrix.
        """
        out = []

        for i in range(0, len(frames), self.batch):
            batch = [f for f in frames[i:i+self.batch] if f is not None]
            if not batch:
                continue

            # Convert from BGR (OpenCV) to RGB for PyTorch model
            batch_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch]

            # Apply transforms
            batch_t = torch.stack([self.tf(f) for f in batch_rgb]).to(self.device)

            # Extract features
            feats = self.model(batch_t).cpu().numpy()
            out.append(feats)

        if not out:
            raise RuntimeError("❌ No valid frames encoded. Check input video path or decoding.")

        feats = np.vstack(out)

        # L2 normalization
        feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9
        return feats
