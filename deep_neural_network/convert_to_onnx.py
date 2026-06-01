#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch
import torch.nn as nn

class SimpleLocalizationCNN(nn.Module):
    def __init__(self, input_channels=2):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, 5, 2, 2),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 5, 2, 2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

def convert(pth_path, onnx_path, height=384, width=384):
    model = SimpleLocalizationCNN(input_channels=2)
    state_dict = torch.load(pth_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 2, height, width)
    
    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                     'output': {0: 'batch_size'}}
    )
    
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path)
    test_input = np.random.randn(1, 2, height, width).astype(np.float32)
    outputs = session.run(None, {'input': test_input})
    print(f"Success! Output shape: {outputs[0].shape}")

if __name__ == '__main__':
    pth = sys.argv[1] if len(sys.argv) > 1 else 'localization_model.pth'
    onnx = sys.argv[2] if len(sys.argv) > 2 else 'localization_model.onnx'
    convert(pth, onnx)