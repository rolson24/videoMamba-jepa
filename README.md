# Video Mamba JEPA

Official PyTorch codebase for VideoMamba trained with V-JEPA by Raif Olson. This repository borrows heavily from the [V-JEPA repository](https://github.com/facebookresearch/jepa) and the [VideoMamba repository](https://github.com/OpenGVLab/VideoMamba/tree/main)



Our version of VideoMamba is trained by passively watching video pixels from large video datasets like SSv2 and the Kinetics dataset in an atttempt to scale VideoMamba models to the size of SOTA ViT models without requiring massive amounts of training time on large datasets to produce good representations.

## Method
V-JEPA pretraining is based solely on an unsupervised feature prediction objective, and does not utilize pretrained image encoders, text, negative examples, human annotations, or pixel-level reconstruction.



## Code Structure

**Config files:**
All experiment parameters are specified in config files (as opposed to command-line arguments). See the [configs/](configs/) directory for example config files. Note, before launching an experiment, you must update the paths in the config file to point to your own directories, indicating where to save the logs and checkpoints and where to find the training data.


```
.
├── app                       # the only place where training loops are allowed
│   ├── vjepa                 #   Video JEPA pre-training
│   ├── main_distributed.py   #   entrypoint for launching app on slurm cluster
│   └── main.py               #   entrypoint for launching app locally on your machine for debugging
├── evals                     # the only place where evaluation of 'apps' are allowed
│   ├── image_classification  #   training an attentive probe for image classification with frozen backbone
│   ├── video_classification  #   training an attentive probe for video classification with frozen backbone
│   ├── main_distributed.py   #   entrypoint for launching distributed evaluations on slurm cluster
│   └── main.py               #   entrypoint for launching evaluations locally on your machine for debugging
├── src                       # the package
│   ├── datasets              #   datasets, data loaders, ...
│   ├── models                #   model definitions
│   ├── masks                 #   mask collators, masking utilities, ...
│   └── utils                 #   shared utilities
└── configs                   # the only place where config files are allowed (specify experiment params for app/eval runs)
    ├── evals                 #   configs for launching vjepa frozen evaluations
    └── pretrain              #   configs for launching vjepa pretraining

```

## Data preparation

### Video Datasets
V-JEPA pretraining and evaluations work with many standard video formats.
To make a video dataset compatible with the V-JEPA codebase, you simply need to create a `.csv` file with the following format and then specify the path to this CSV file in your config.
```
/absolute_file_path.[mp4, webvid, etc.] $integer_class_label
/absolute_file_path.[mp4, webvid, etc.] $integer_class_label
/absolute_file_path.[mp4, webvid, etc.] $integer_class_label
...
```
Since V-JEPA is entirely unsupervised, the pretraining code will disregard the `$integer_class_label` in the CSV file.
Thus, feel free to put a random value in this column.
However, if you wish to run a supervised video classification evaluation on your video dataset, you must replace ```$integer_class_label``` with the ground truth label for each video.

### Image Datasets
We use the standard PyTorch ```ImageFolder``` class in our image classification evals.
Thus, to set up an image dataset for the image classification evaluation, first create a directory to store your image datasets ```$your_directory_containing_image_datasets```.
Next, download your image datasets into this directory in a format compatible with [PyTorch ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html).

For example, suppose we have a directory called ``my_image_datasets``. We would then download our image datasets into this directory so that we end up with the following file tree
```
.
└── /my_image_datasets/                # where we store image datasets
    ├── places205/121517/pytorch/      #   Places205
    │   └── [...]
    ├── iNaturalist-2021/110421/       #   iNaturalist21
    │   └── [...]
    ├── [...]                          #   Other Image Datasets
    │   └── [...]
    └── imagenet_full_size/061417/     #   ImageNet1k
        └── train
        │   ├── $class_1
        │   │    ├── xxx.[png, jpeg, etc.]
        │   │    ├── [...]
        │   │    └── xxz.[png, jpeg, etc.]
        │   ├── [...]
        │   └── $class_n
        │       ├── abc.[png, jpeg, etc.]
        │       ├── [...]
        │       └── abz.[png, jpeg, etc.]
        └── val
            ├── $class_1
            │    ├── xxx.[png, jpeg, etc.]
            │    ├── [...]
            │    └── xxz.[png, jpeg, etc.]
            ├── [...]
            └── $class_n
                ├── abc.[png, jpeg, etc.]
                ├── [...]
                └── abz.[png, jpeg, etc.]
```


## Launching V-JEPA pretraining

### Local training
If you wish to debug your code or setup before launching a distributed training run, we provide the functionality to do so by running the pretraining script locally on a multi-GPU (or single-GPU) machine, however, reproducing our results requires launching distributed training.

The single-machine implementation starts from the [app/main.py](appmain.py), which parses the experiment config file and runs the pretraining locally on a multi-GPU (or single-GPU) machine.
For example, to run V-JEPA pretraining on GPUs "0", "1", and "2" on a local machine using the config [configs/pretrain/vitl16.yaml](configs/pretrain/vitl16.yaml), type the command:
```bash
python -m app.main \
  --fname configs/pretrain/videomambaT16.yaml \
  --devices cuda:0 cuda:1 cuda:2
```

### Distributed training
To launch a distributed training run, the implementation starts from [app/main_distributed.py](app/main_distributed.py), which, in addition to parsing the config file, also allows for specifying details about distributed training. For distributed training, we use the popular open-source [submitit](https://github.com/facebookincubator/submitit) tool and provide examples for a SLURM cluster.

For example, to launch a distributed pre-training experiment using the config [configs/pretrain/vitl16.yaml](configs/pretrain/vitl16.yaml), type the command:
```bash
python -m app.main_distributed \
  --fname configs/pretrain/videomambaT16.yaml \
  --folder $path_to_save_stderr_and_stdout \
  --partition $slurm_partition
```

## Launching Evaluations

### Local training
If you wish to debug your eval code or setup before launching a distributed training run, we provide the functionality to do so by running the evaluation script locally on a multi-GPU (or single-GPU) machine, however, reproducing the full eval would require launching distributed training.
The single-machine implementation starts from the [eval/main.py](eval/main.py), which parses the experiment config file and runs the eval locally on a multi-GPU (or single-GPU) machine.

For example, to run SSv2 Video classification on GPUs "0", "1", and "2" on a local machine using the config [configs/eval/videomambaT16_ssv2_16x2x3.yaml](configs/eval/videomambaT16_ssv2_16x2x3.yaml), type the command:
```bash
python -m evals.main \
  --fname configs/eval/videomambaT16_ssv2_16x2x3.yaml \
  --devices cuda:0 cuda:1 cuda:2
```


### Distributed training
To launch a distributed evaluation run, the implementation starts from [eval/main_distributed.py](eval/main_distributed.py), which, in addition to parsing the config file, also allows for specifying details about distributed training. For distributed training, we use the popular open-source [submitit](https://github.com/facebookincubator/submitit) tool and provide examples for a SLURM cluster.

For example, to launch a distributed ImageNet image classification experiment using the config [configs/eval/videomambaT16_ssv2_16x2x3.yaml](configs/eval/videomambaT16_ssv2_16x2x3.yaml), type the command:
```bash
python -m evals.main_distributed \
  --fname configs/eval/videomambaT16_ssv2_16x2x3.yaml \
  --folder $path_to_save_stderr_and_stdout \
  --partition $slurm_partition
```


---

### Setup

Run:
```bash
conda create -n jepa python=3.9 pip
conda activate jepa
python setup.py install
```

## License
See the [LICENSE](./LICENSE) file for details about the license under which this code is made available.

