This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	--tags=generate-run-cmds,inference,_populate-readme \
	--model=resnet50 \
	--implementation=reference \
	--device=cuda \
	--backend=tf \
	--category=edge \
	--division=closed \
	--quiet \
	--results_dir=/home/arjun/results_dir \
	--skip_submission_generation=yes \
	--execution-mode=valid \
	--power=yes \
	--adr.mlperf-power-client.power_server=192.168.0.15 \
	--adr.mlperf-power-client.port=4950 \
	--scenario=Offline
```
## Dependent CM scripts 


1.  `cm run script --tags=detect,os`


2.  `cm run script --tags=get,sys-utils-cm`


3.  `cm run script --tags=get,python`


4.  `cm run script --tags=get,mlcommons,inference,src`


5.  `cm run script --tags=get,dataset-aux,imagenet-aux`

## Dependent CM scripts for the MLPerf Inference Implementation


1. `cm run script --tags=detect,os`


2. `cm run script --tags=detect,cpu`


3. `cm run script --tags=get,sys-utils-cm`


4. `cm run script --tags=get,python`


5. `cm run script --tags=get,cuda,_cudnn`


6. `cm run script --tags=get,generic-python-lib,_tensorflow`


7. `cm run script --tags=get,ml-model,image-classification,resnet50,_fp32,_tf,raw`


8. `cm run script --tags=get,dataset,image-classification,imagenet,preprocessed,_default,_full,_NHWC`


9. `cm run script --tags=get,dataset-aux,image-classification,imagenet-aux`


10. `cm run script --tags=generate,user-conf,mlperf,inference`


11. `cm run script --tags=get,loadgen`


12. `cm run script --tags=get,mlcommons,inference,src`


13. `cm run script --tags=get,generic-python-lib,_opencv-python`


14. `cm run script --tags=get,generic-python-lib,_numpy`


15. `cm run script --tags=get,generic-python-lib,_pycocotools`