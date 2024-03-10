import PIL.Image as pil
import argparse
import glob
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import onnxruntime as ort
import os
import sys
import torch
import torch.onnx
from torchsummary import summary
from torchvision import transforms, datasets

def main():

    device = torch.device("cpu")

    # Load image and preprocess
    input_image = pil.open("test_image.jpg").convert('RGB')
    original_width, original_height = input_image.size
    input_image = input_image.resize((434, 392), pil.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)


    #run onnx
    encoder_path = "/home/devon/ws/Depth-Anything-ONNX/weights/depth_anything_vits14.encoder.onnx"
    decoder_path = "/home/devon/ws/Depth-Anything-ONNX/weights/depth_anything_vits14.decoder.onnx"

    encoder_sess = ort.InferenceSession(encoder_path, providers=ort.get_available_providers())
    decoder_sess = ort.InferenceSession(decoder_path, providers=ort.get_available_providers())

    input_name = encoder_sess.get_inputs()[0].name
    features_name = decoder_sess.get_inputs()[0].name

    features = encoder_sess.run(None, {input_name: input_image.numpy()})
    outputs = decoder_sess.run(None, {features_name: np.array(features[0])})

    #test image

    

    disp = torch.FloatTensor(outputs[0])#.unsqueeze(1)
    print("output size: ", disp.shape)
    print(disp[0][0][200])

    disp_resized = torch.nn.functional.interpolate(
        disp, (original_height, original_width), mode="bilinear", align_corners=False)

    # Saving colormapped depth image
    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='binary')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)
    im.save("onnx_disp.jpg")

if(__name__ == "__main__"):
    main()