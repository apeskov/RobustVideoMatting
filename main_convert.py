import os
import torch
from torch import nn
from model import MattingNetwork


class TraceWrapper(nn.Module):
    def __init__(self, model, downsample_ratio):
        super().__init__()
        self.model = model
        self.downsample_ratio = downsample_ratio

    def forward(self, *inp):
        return self.model(*inp, self.downsample_ratio)

    def fuse_model(self):
        self.model.backbone.fuse_model()


def load_rvm(model_path, downsample_ratio=None, frame_size=None, trace=False):
    variant = "mobilenetv3" if "mobilenetv3" in model_path else "resnet50"
    model = MattingNetwork(variant=variant)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    model = TraceWrapper(model, downsample_ratio)
    input_shapes = [None]*5

    if frame_size:
        frame = torch.randn(1, 3, *frame_size)
        rec = [None] * 4
        fgr, pha, *rec = model(frame, *rec)
        input_shapes = [list(fgr.shape)]
        input_shapes += [list(r.shape) for r in rec]

    if trace:
        assert input_shapes
        frame = torch.randn(1, 3, *frame_size)
        rec = [None] * 4

        fgr, pha, *rec = model(frame, *rec)  # Just to define shape of states
        model = torch.jit.trace(model, [fgr, *rec])

    model.input_shapes = input_shapes
    return model


def save_as_script(model, out_name):
    # Store shape info as metadata
    shapes_str = ""
    input_stub = []
    for shape in model.input_shapes:
        assert shape
        shapes_str += ",".join((str(d) for d in shape)) + "\n"
        input_stub.append(torch.randn(shape))

    model = torch.jit.trace(model, input_stub)
    torch.jit.save(torch.jit.script(model), out_name, _extra_files={'shapes.txt': shapes_str})


def main():
    models_dir = "models_ref"
    result_dir = "models_trace"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    frame_size = [1080, 1920]
    downsample_ratio = 0.25

    for name in ["rvm_mobilenetv3", "rvm_resnet50"]:
        ref_model_path = f"{models_dir}/{name}.pth"
        model = load_rvm(ref_model_path, frame_size=frame_size, downsample_ratio=downsample_ratio, trace=True)

        save_as_script(model, out_name=f"{result_dir}/{name}_fp32_trace.torchscript")
        model.model.half()
        save_as_script(model, out_name=f"{result_dir}/{name}_fp16_trace.torchscript")


if __name__ == '__main__':
    main()
