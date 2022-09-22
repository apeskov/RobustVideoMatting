import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from inference_utils import VideoReader, VideoWriter

# some files form original RVM footage storage
# https://drive.google.com/drive/folders/1VFnWwuu-YXDKG-N6vcjK_nL7YZMFapMU
DATASET_DIR = "/Users/apeskov/Downloads"
VALIDATION_DATASET = [
    ("es2.mp4", 60),
    ("codylexi.mp4", 60)
]


def evaluate(model, in_file=None, out_file=None, shapes=None, frame_limit=100500):
    frame_shape = shapes[0]
    input_transform = transforms.Compose([
        transforms.Resize(frame_shape[2:]),
        transforms.ToTensor()
    ])

    reader = VideoReader(in_file, transform=input_transform)
    writer = VideoWriter(out_file, frame_rate=30) if out_file else None

    bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1)  # Green background.
    rec = [torch.empty(*shape) for shape in shapes[1:]]

    count = 0

    with torch.no_grad():
        for src in DataLoader(reader):
            fgr, pha, *rec = model(src, *rec)  # Cycle the recurrent states.
            print("*")
            if writer:
                writer.write(fgr * pha + bgr * (1 - pha))
            count += 1
            if count > frame_limit:
                break


def load_rvm(model_file):
    extra_files = {"shapes.txt": ""}
    model = torch.jit.load(model_file, _extra_files=extra_files)
    model.eval()

    input_shapes = []
    for line in extra_files["shapes.txt"].decode("utf-8").splitlines():
        input_shapes.append([int(d) for d in line.split(",")])

    return model, input_shapes


def main():
    result_dir = "verification_outs"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for name in [
        # "rvm_resnet50_fp32_trace",
        # "rvm_resnet50_fp16_trace",
        "rvm_resnet50_int8",
        # "rvm_mobilenetv3_fp32_trace",
        # "rvm_mobilenetv3_fp16_trace",
        "rvm_mobilenetv3_int8"
    ]:
        model_dir = "models_ref"
        model, input_shapes = load_rvm(f"{model_dir}/{name}.torchscript")
        for video_file, frame_limit in VALIDATION_DATASET:
            file_name = video_file.split(".")[0]
            in_video_file_path = f"{DATASET_DIR}/{video_file}"
            out_video_file_path = f"{result_dir}/{file_name}_{name}.mp4"

            evaluate(model,
                     in_file=in_video_file_path,
                     out_file=out_video_file_path,
                     shapes=input_shapes,
                     frame_limit=60)


if __name__ == '__main__':
    main()
