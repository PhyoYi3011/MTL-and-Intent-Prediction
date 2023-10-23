# Unveiling the Dynamics of Crisis Events: Sentiment and Emotion Analysis via Multi-Task Learning with Attention Mechanism and Subject-based Intent Prediction

## Multi-Task Learning Model

Fine-tuned MTL models (BERT, RoBERTa, BERT) can be downloaded [here](https://drive.google.com/drive/folders/1xNaOG-4VS2emW2N7IStLm-E2EzAltbde?usp=sharing).

## COMET-ATOMIC 2020 Model

COMET-ATOMIC 2020 models (BART, GPT2-XL) can be downloaded [here](https://github.com/allenai/comet-atomic-2020).


## Prerequisites to Run Models

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
Run `pip install -r requirements.txt` to install requirements for your Python instance.Our codebases is on Python 3.

Make sure all the prerequisites are installed and set up correctly. 

Follow the file directory structure to store the respective files. 

# Running MTL Models

Run `python mtl/bertweet.cpython-38.pyc`





