import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel

dialogpt_model = GPT2LMHeadModel.from_pretrained("/content/drive/MyDrive/mangrove_finetuned_DialoGPT")
dialogpt_model.eval()

max_sequence_length = 1024
dummy_input = torch.zeros((1, max_sequence_length), dtype=torch.long)
onnx_path = "dialogpt_model.onnx"
torch.onnx.export(dialogpt_model, dummy_input, onnx_path, verbose=True)

import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
max_workspace_size = 15

builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(max_workspace_size)
parser = trt.OnnxParser(network, TRT_LOGGER)

builder.max_workspace_size = 1 << 30
builder.max_batch_size = 1

with open(onnx_path, 'rb') as model_file:
    if not parser.parse(model_file.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))

engine = builder.build_cuda_engine(network)


h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)


input_data = np.random.random((1, max_sequence_length)).astype(np.float32)
np.copyto(h_input, input_data.ravel())


with engine.create_execution_context() as context:
    cuda.memcpy_htod(d_input, h_input)
    context.execute(1, [int(d_input), int(d_output)])
    cuda.memcpy_dtoh(h_output, d_output)


output = np.reshape(h_output, engine.get_binding_shape(1))
print(output)