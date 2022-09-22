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


def load_rvm(model_path, downsample_ratio=None, frame_size=None):
    variant = "mobilenetv3" if "mobilenetv3" in model_path else "resnet50"
    model = MattingNetwork(variant=variant)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return TraceWrapper(model, downsample_ratio)


def get_input_shapes(model, frame_size):
    frame = torch.randn(1, 3, *frame_size)
    rec = [None] * 4

    fgr, pha, *rec = model(frame, *rec)  # Just to define shape of states
    rec_shapes = [list(r.shape) for r in rec]

    return [[1, 3, *frame_size]] + rec_shapes


def save_as_script(model, out_name, frame_size):
    input_shapes = get_input_shapes(model, frame_size)

    # Store shape info as metadata
    shapes_str = ""
    input_stub = []
    for shape in input_shapes:
        shapes_str += ",".join((str(d) for d in shape)) + "\n"
        input_stub.append(torch.randn(shape))

    model = torch.jit.trace(model, input_stub)
    torch.jit.save(torch.jit.script(model), out_name, _extra_files={'shapes.txt': shapes_str})


def convert(model_path, model_name, shape, downsample_ratio):
    model = load_rvm(model_path, frame_size=shape, downsample_ratio=downsample_ratio)
    save_as_script(model, out_name=f"{model_name}_fp32_trace", frame_size=shape)
    model.model.half()
    save_as_script(model, out_name=f"{model_name}_fp16_trace", frame_size=shape)


def main():
    models_dir = "models_ref"
    for name in ["rvm_mobilenetv3", "rvm_resnet50"]:
        ref_model_path = f"{models_dir}/{name}.pth"
        convert(ref_model_path, model_name=name, shape=[1080, 1920], downsample_ratio=0.25)


if __name__ == '__main__':
    main()
