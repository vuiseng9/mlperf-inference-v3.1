import time
import os
import sys
import torch
import logging
from pathlib import Path

import intel_extension_for_pytorch as ipex
from typing import Optional, Tuple, Union

USE_TPP=int(os.environ.get("USE_TPP", "0")) == 1
print(f'Use TPP: {USE_TPP}')

torch._C._jit_set_texpr_fuser_enabled(False)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("BACKEND")

class ResnetBackend(object):

    def __init__(self, model_checkpoint=None, precision="int8", quantized_model=None):

        self.model_checkpoint = model_checkpoint
        self.precision = precision
        self.amp_enabled = False
        self.amp_dtype = None

        self.quantized_model = quantized_model

        if self.precision=="int8":
            if self.quantized_model is None:
                print("Running int8 requires a 'quantized_model' path")
                sys.exit(1)
            elif not os.path.isfile(self.quantized_model):
                print("Path to quantized model {} not found".format(self.quantized_model))
                sys.exit(1)
        
    def loadModel(self):
        """ Loads the pretrained model """
        # self.model = torch.jit.load(self.model_checkpoint)
        if self.precision in ["int8"]:
            torch._C._jit_set_texpr_fuser_enabled(False)
            # self.model = self.model.to(memory_format=torch.channels_last)


            self.model = torch.jit.load(self.quantized_model)
            self.model = torch.jit.freeze(self.model.eval())
            print("Load successful")
            # setattr(self.model, "trace_graph", self.int8_model)
        else:
            raise ValueError(f"Precision {self.precision} unsupported for resnet50")
        # if self.precision=="bf16":
        #     if USE_TPP == True:
        #         from tpp_pytorch_extension.llm.fused_gptj_infer import OptimizeModelForGPTJ
        #         import tpp_pytorch_extension as tpx
        #         OptimizeModelForGPTJ(self.model, dtype=torch.bfloat16, device='cpu')
        #         self.model = tpx.llm.llm_common.jit_trace_model(self.model, self.tokenizer, self.generate_kwargs["num_beams"])
        #     else:
        #         self.model = ipex.optimize(self.model,
        #                 dtype=torch.bfloat16,
        #                 inplace=True,
        #                 concat_linear=False
        #                 )

    def predict(self, input_batch):
        """ Runs inference on 'input_batch' """
        enable_autocast = self.precision in ["bf16", "mix", "int4_bf16_mixed"]
        with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(enabled=enable_autocast, dtype=torch.bfloat16):
            outputs = self.model(input_batch)
        return outputs


