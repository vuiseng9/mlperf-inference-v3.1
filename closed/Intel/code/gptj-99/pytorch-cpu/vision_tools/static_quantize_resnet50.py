# Notes:
# inherit from https://github.com/intel/intel-extension-for-pytorch/blob/master/examples/cpu/inference/python/int8_calibration_static.py
# commit id e4f32b8
# modify channel last

import torch
#################### code changes ####################  # noqa F401
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import prepare, convert
######################################################  # noqa F401

##### Example Model #####  # noqa F401
import torchvision.models as models
model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
model = model.to(memory_format=torch.channels_last)
model.eval()
data = torch.rand(1, 3, 224, 224)
#########################  # noqa F401

qconfig_mapping = ipex.quantization.default_static_qconfig_mapping
# Alternatively, define your own qconfig_mapping:
# from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig, QConfigMapping
# qconfig = QConfig(
#        activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
#        weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
# qconfig_mapping = QConfigMapping().set_global(qconfig)
prepared_model = prepare(model, qconfig_mapping, example_inputs=data, inplace=False)

##### Example Dataloader #####  # noqa F401
import torchvision
DOWNLOAD = True
DATA = 'datasets/cifar10/'

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = torchvision.datasets.CIFAR10(
    root=DATA,
    train=True,
    transform=transform,
    download=DOWNLOAD,
)
calibration_data_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=128
)

with torch.no_grad():
    for batch_idx, (d, target) in enumerate(calibration_data_loader):
        print(f'calibrated on batch {batch_idx} out of {len(calibration_data_loader)}')
        d = d.to(memory_format=torch.channels_last)
        prepared_model(d)
        if batch_idx > 4:
            break
##############################  # noqa F401

converted_model = convert(prepared_model)
with torch.no_grad():
    traced_model = torch.jit.trace(converted_model, data)
    traced_model = torch.jit.freeze(traced_model)

traced_model.save("quantized_resnet50.pt")

print("Execution finished")
