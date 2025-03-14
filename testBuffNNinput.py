import torch
from torch import nn
import onnx
from onnxsim import simplify
import blobconverter

shape = (1,1)
X = torch.ones(shape, dtype=torch.float16)

class testBuff(nn.Module):
    def forward(self, inp):                        
        out = inp  
        return out

filename = "testBuff"
onnx_file = filename + ".onnx"

torch_model = testBuff()

print(f"Writing to {onnx_file}")
torch.onnx.export(
    torch_model,
    X,
    onnx_file,
    opset_version=13,
    do_constant_folding=True,
    input_names = ['inp'], # Optional
    output_names = ['out'], # Optional
)

# Use onnx-simplifier to simplify the onnx model
print("Simplifying the ONNX model")
onnx_simplified_path = filename + "_simplified.onnx"
onnx_model = onnx.load(onnx_file)
model_simp, check = simplify(onnx_model)
onnx.save(model_simp, onnx_simplified_path)

# Use blobconverter to convert onnx->IR->blob
blobconverter.from_onnx(
    model=onnx_simplified_path,
    data_type="FP16",
    shaves=6,
    use_cache=False,
    output_dir=".",
    optimizer_params=[]
)