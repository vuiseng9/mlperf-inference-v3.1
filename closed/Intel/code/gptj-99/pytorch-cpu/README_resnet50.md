### Download & Prepare ImageNet Dataset
The current approach is leveraging torchvision as input loader. Steps differs from MLPerf references.
```bash
# run at this level
# mlperf-inference-v3.1/closed/Intel/code/gptj-99/pytorch-cpu
bash download_imagenet.sh
```
```ILSVRC2012_img_val``` should contain 50K JPEGs. Then we need to organize per torchvision dataset scheme, i.e. all images are put in the same folder, one folder per class.
```bash
cd ILSVRC2012_img_val
cp vision_tools/valprep.sh .
./valprep.sh # be patient, this may take few minutes with out any traces.
```
check if ```ILSVRC2012_img_val/val``` exists, with 1000 folders (imagenet validation has 1000 classes, 50 images per class)

### Prepare 8-bit resnet
```bash
python vision_tools/static_quantize_resnet50.py
```

### Run
```bash
source setup_env.sh
bash rn50-run_offline_accuracy_50_examples.sh
```

Opens
* do the current get sample through torch sampler create overhead to benchmarking? most likely not, just double confirm.
* the current implementation is not efficient where each consumer create a copy of torch tensor in every new launch. The implication of this is the initial (setup) stage of benchmarking will be long.
* verify channel_last performance

