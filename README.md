# Unveiling the Dynamics of Crisis Events: Sentiment and Emotion Analysis via Multi-Task Learning with Attention Mechanism and Subject-based Intent Prediction

## Multi-Task Learning Model

Fine-tuned MTL models (BERT, RoBERTa, BERT) can be downloaded [here](https://drive.google.com/drive/folders/1xNaOG-4VS2emW2N7IStLm-E2EzAltbde?usp=sharing).

## COMET-ATOMIC 2020 Model

COMET-ATOMIC 2020 models (BART, GPT2-XL) can be downloaded [here](https://github.com/allenai/comet-atomic-2020).


## Pre-requisites to Run Models

NVIDIA CUDA toolkit. Refer to this installation [guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) for Linux or [guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) for Microsoft Windows.

NVIDIA container toolkit. Refer to this installation [guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

To find out if pytorch is using GPU, run below python script.
  * import torch
  * torch.cuda.is_available()
  * torch.cuda.current_device()
      * “0” if “cuda:0” is used.
      * “1” if “cuda:1” is used.

## Directory Overview
`dataset`: Contains two folders: `input` and `output`. `input` contains sample testing dataset needed to run MTL and COMET-ATOMIC models. `output` contains output files generated after running MTL and COMET-ATOMIC models

`mtl`: Contains compiled python files to run BERT, RoBERTa, BERTweet models

`comet-atomic`: Contains compiled python files to run BART and GPT2-XL models


## Setup
Run `pip install -r requirements.txt` to install requirements for your Python instance.Our codebases is on Python 3.8.

Make sure all the pre-requisites are installed and set up correctly. 

Follow the exact file directory structure to store all the files. 

# Running MTL Models

Run `python mtl/bertweet.pyc` 

For the input prompt `Enter GPU device name:`, enter either `cuda:0` or `cuda:1` (based on what you get from setting up the pre-requisites). 

For the input prompt `Enter BERTweet model path:`, enter the file path of `mtl_bertweet.pt` (e.g. `model/mtl_bertweet.pt`)

Output file can be found at `dataset/output/mtl/bertweet_output.csv`

# Running COMET-ATOMIC Models

Run `python comet-atomic/BART.pyc` 

For the input prompt `Enter GPU device name:`, enter either `cuda:0` or `cuda:1` (based on what you get from setting up the pre-requisites). 

For the input prompt `Enter BART model path:`, enter the file path of `comet-atomic_2020_BART` (e.g. `model/comet-atomic_2020_BART`)

Output file can be found at `dataset/output/comet-atomic/BART_output.csv`












