import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from inference_utils import VideoReader
from main_convert import save_as_script, load_rvm

# some files form original RVM footage storage
# https://drive.google.com/drive/folders/1VFnWwuu-YXDKG-N6vcjK_nL7YZMFapMU
DATASET_DIR = "streams"
CALIBRATION_DATASET = [
    ("es2.mp4", 30),
    ("codylexi.mp4", 30)
]


def evaluate(model, video_shape=None, dataset=CALIBRATION_DATASET):
    if video_shape:
        input_transform = transforms.Compose([
            transforms.Resize(video_shape),
            transforms.ToTensor()
        ])
    else:
        input_transform = transforms.ToTensor()

    for video_file, frame_limit in dataset:
        video_file_path = f"{DATASET_DIR}/{video_file}"
        reader = VideoReader(video_file_path, transform=input_transform)
        rec = [None] * 4
        cur_frame = 0

        with torch.no_grad():
            for src in DataLoader(reader):
                fgr, pha, *rec = model(src, *rec)
                print("*")
                cur_frame = cur_frame + 1
                if cur_frame >= frame_limit:
                    break


def quantize(model_path, out_name, frame_size=[1080, 1920], downsample_ratio=0.25):
    model = load_rvm(model_path, downsample_ratio=downsample_ratio, frame_size=frame_size)

    # Fuse model. Prerequisite
    model.fuse_model()

    # Quantization config (default)
    model.model.backbone.qconfig = torch.quantization.default_qconfig
    torch.quantization.prepare(model.model.backbone, inplace=True)

    # Collect statistic
    evaluate(model, video_shape=frame_size)

    # Convert to quantized version
    torch.quantization.convert(model, inplace=True)

    save_as_script(model, out_name=out_name)


def main():
    models_ref_dir = "models_ref"
    result_dir = "models_trace"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for name in ["rvm_resnet50", "rvm_mobilenetv3"]:
        quantize(f"{models_ref_dir}/{name}.pth", out_name=f"{result_dir}/{name}_int8_trace.torchscript")


if __name__ == '__main__':
    main()
