import argparse

import math
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
from onnx import load_model, save_model
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

from depth_anything.dpt import DPT_DINOv2
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


class Encoder(nn.Module):
    def __init__(self, main_model, use_clstoken):

        super(Encoder, self).__init__()
        self.main_model = main_model
        self.use_clstoken = use_clstoken


    def forward(self, x, get_joints=False): #just run the half of DPT_DINOv2 features responsible for the encoder

        features = self.main_model.pretrained.get_intermediate_layers(
            x, 4, return_class_token=True
        )

        flat_features = []
        joints = []
        dims = []
        
        for i, x in enumerate(features):

            if self.use_clstoken:
                x, cls_token = x[0], x[1]

                t = torch.flatten(cls_token)
                flat_features.append(t)
                joints.append(list(t.shape)[0])
                dims.append(list(cls_token.shape))
                
            else:
                x = x[0]

            t = torch.flatten(x)
            flat_features.append(t)
            joints.append(list(t.shape)[0])
            dims.append(list(x.shape))

        grouped_features = torch.cat(tuple(flat_features))

        if( get_joints):
            return grouped_features, joints, dims
        else:
            return grouped_features
    
class Decoder(nn.Module):
    def __init__(self, main_model, h, w, dims, joints, use_clstoken):

        super(Decoder, self).__init__()
        self.main_model = main_model
        self.h = h
        self.w = w
        self.dims = dims
        self.joints = joints
        self.use_clstoken = use_clstoken



    def forward(self, features): #just run the half of DPT_DINOv2 features responsible for the decoder

        view_tensors = torch.split(features, self.joints)

        new_features = []
        if(self.use_clstoken):
            for i in range(0, int(len(self.joints)), 2):

                
                token = torch.reshape(view_tensors[i], self.dims[i])
                x = torch.reshape(view_tensors[i+1], self.dims[i+1])

                f = (x, token)
                new_features.append(f)
        else:
            assert False # TODO not sure how yet

        new_features = tuple(new_features)

        patch_h, patch_w = self.h // 14, self.w // 14

        depth = self.main_model.depth_head(new_features, patch_h, patch_w)
        depth = F.interpolate(depth, size=(self.h, self.w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)

        return depth
    



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=["s", "b", "l"],
        required=True,
        help="Model size variant. Available options: 's', 'b', 'l'.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        required=False,
        help="Path to save the ONNX model.",
    )
    

    return parser.parse_args()


def export_onnx(model: str, height: int, width: int, output: str = None):
    # Handle args
    if output is None:
        output = f"weights/depth_anything_vit{model}14.onnx"

    encoder_output = output.replace(".onnx", ".encoder.onnx")
    decoder_output = output.replace(".onnx", ".decoder.onnx")    

    # Device for tracing (use whichever has enough free memory)
    device = torch.device("cpu")#"cuda" if torch.cuda.is_available() else "cpu")

    # Sample image for tracing (dimensions don't matter)
    image, _ = load_image("assets/1024x320.png", height, width)
    image = torch.from_numpy(image).to(device)

    # Load model params
    if model == "s":
        depth_anything = DPT_DINOv2(
            encoder="vits", features=64, out_channels=[48, 96, 192, 384]
        )
    elif model == "b":
        depth_anything = DPT_DINOv2(
            encoder="vitb", features=128, out_channels=[96, 192, 384, 768]
        )
        

    else:  # model == "l"
        depth_anything = DPT_DINOv2(
            encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]
        )
    
    encoder = Encoder(depth_anything, True)

    depth_anything.to(device).load_state_dict(
        torch.hub.load_state_dict_from_url(
            f"https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vit{model}14.pth",
            map_location="cpu",
        ),
        strict=True,
    )
    depth_anything.eval()
    encoder.eval()


    torch.onnx.export(
        encoder,
        image,
        encoder_output,
        input_names=["image"],
        output_names=["features"],
        opset_version=11,
        # dynamic_axes={
        #     "image": {2: "height", 3: "width"},
        #     "depth": {2: "height", 3: "width"},
        # },
    )

    features, joints, dims = encoder.forward(image, get_joints=True)
    # exit()  

    decoder = Decoder(depth_anything, height, width, dims, joints, True)
    decoder.eval()

    torch.onnx.export(
        decoder,
        features,
        decoder_output,
        input_names=["features"],
        output_names=["depth"],
        opset_version=11,
        # dynamic_axes={
        #     "image": {2: "height", 3: "width"},
        #     "depth": {2: "height", 3: "width"},
        # },
    )

    og_test = depth_anything(image)
    f = encoder(image)
    split_test = decoder(f)

    print("split == original: ", torch.equal(og_test, split_test))

    save_model(
        SymbolicShapeInference.infer_shapes(load_model( encoder_output), auto_merge=True),
        output,
    )

    save_model(
        SymbolicShapeInference.infer_shapes(load_model( decoder_output), auto_merge=True),
        output,
    )



def load_image(filepath, h, w) -> tuple[np.ndarray, tuple[int, int]]:

    transform = Compose(
    [
        Resize(
            width=w,
            height=h,
            resize_target=False,
            keep_aspect_ratio=False,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ]
)


    image = cv2.imread(filepath)  # H, W, C
    orig_shape = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    image = transform({"image": image})["image"]  # C, H, W
    image = image[None]  # B, C, H, W
    return image, orig_shape


if __name__ == "__main__":
    args = parse_args()
    export_onnx(**vars(args))
