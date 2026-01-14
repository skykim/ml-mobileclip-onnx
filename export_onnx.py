import torch
import open_clip
from mobileclip.modules.common.mobileone import reparameterize_model
import onnx
import os
import warnings
from huggingface_hub import hf_hub_download

output_folder = "onnx"
os.makedirs(output_folder, exist_ok=True)

checkpoint_path = hf_hub_download(repo_id="apple/MobileCLIP2-S2", filename="mobileclip2_s2.pt")
model, _, _ = open_clip.create_model_and_transforms('MobileCLIP2-S2', pretrained=checkpoint_path)

model.eval()
model = reparameterize_model(model)

class VisionModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, pixel_values):
        return self.model.encode_image(pixel_values)

class TextModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_ids):
        return self.model.encode_text(input_ids)

vision_wrapper = VisionModelWrapper(model)
text_wrapper = TextModelWrapper(model)

dummy_pixel_values = torch.randn(1, 3, 256, 256)
dummy_input_ids = torch.randint(0, 49408, (1, 77), dtype=torch.long)

def set_dim_names(onnx_model, dim_map):
    for input_tensor in onnx_model.graph.input:
        if input_tensor.name in dim_map:
            for dim_idx, name in dim_map[input_tensor.name].items():
                input_tensor.type.tensor_type.shape.dim[dim_idx].dim_param = name
    for output_tensor in onnx_model.graph.output:
        if output_tensor.name in dim_map:
            for dim_idx, name in dim_map[output_tensor.name].items():
                output_tensor.type.tensor_type.shape.dim[dim_idx].dim_param = name

def export_and_merge(torch_model, dummy_input, output_path, input_names, output_names, dynamic_axes, explicit_dim_names):
    temp_path = output_path.replace(".onnx", "_temp.onnx")
    
    torch.onnx.export(
        torch_model,
        (dummy_input,),
        temp_path,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes
    )
    
    onnx_model = onnx.load(temp_path, load_external_data=True)
    set_dim_names(onnx_model, explicit_dim_names)
    onnx.save(onnx_model, output_path)
    
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    possible_data_files = [temp_path + ".data", temp_path + "_data"]
    for data_file in possible_data_files:
        if os.path.exists(data_file):
            os.remove(data_file)

vision_output_path = os.path.join(output_folder, "vision_model.onnx")
export_and_merge(
    vision_wrapper,
    dummy_pixel_values,
    vision_output_path,
    input_names=["pixel_values"],
    output_names=["image_embeds"],
    dynamic_axes={"pixel_values": {0: "batch_size"}, "image_embeds": {0: "batch_size"}},
    explicit_dim_names={"pixel_values": {0: "batch_size"}, "image_embeds": {0: "batch_size"}}
)

text_output_path = os.path.join(output_folder, "text_model.onnx")
export_and_merge(
    text_wrapper,
    dummy_input_ids,
    text_output_path,
    input_names=["input_ids"],
    output_names=["text_embeds"],
    dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}, "text_embeds": {0: "batch_size"}},
    explicit_dim_names={"input_ids": {0: "batch_size", 1: "sequence_length"}, "text_embeds": {0: "batch_size"}}
)