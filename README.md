# Unveiling the Dynamics of Crisis Events: Sentiment and Emotion Analysis via Multi-task Learning with Attention Mechanism and Subject-based Intent Prediction

## Multi-task Learning Model (MTL)

Fine-tuned MTL models (BERT, RoBERTa, BERTweet) can be downloaded [here](https://drive.google.com/file/d/1_dDnBZfA5Uvly0Mg3ZmuRFOUk6-uBZ1X/view?usp=share_link). The zipped folder contains `mtl_bert.pt`, `mtl_bertweet.pt`, `mtl_roberta.pt`.

## COMET-ATOMIC 2020 Model

COMET-ATOMIC 2020 models (BART, GPT2-XL) can be downloaded [here](https://drive.google.com/file/d/1ugPVEZiJkDuEFXbt3_jUMeHulu9rPMm8/view?usp=sharing). The zipped folder contains `comet-atomic_2020_BART` and `gpt2xl-comet-atomic-2020`.


## Pre-requisites to Run Models

* NVIDIA CUDA toolkit. Refer to this installation [guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) for Linux or [guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) for Microsoft Windows.

* NVIDIA container toolkit. Refer to this installation [guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).


## Directory Overview
`dataset`: Contains two folders: `input` and `output`. 
   * `input`: Contains sample testing dataset needed to run MTL and COMET-ATOMIC models.
   * `output`: Contains sample output files generated after running MTL and COMET-ATOMIC models.

`mtl`: Contains compiled python files to run BERT, RoBERTa, BERTweet models.

`comet-atomic`: Contains compiled python files to run BART and GPT2-XL models.


## Setup
Make sure all the pre-requisites are installed and set up correctly. 

Run `pip install -r requirements.txt` to install requirements for your Python instance. Our codebases is on Python 3.8.8.

Run `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html` 

To find out if pytorch is using GPU in your environment, run below python script.
  * import torch
  * torch.cuda.is_available()
     * It should return `True`
To find out which GPU device is selected in your environment, run below python script. 
  * torch.cuda.current_device()
      * If it returns `0`, `cuda:0` is used.
      * If it reutnr `1`, `cuda:1` is used.

Follow the exact file directory structure to store all the files. 

## Running MTL Models

Run `python mtl/bertweet.pyc` 

When promped with `Enter GPU device name:`, enter either `cuda:0` or `cuda:1` (based on which GPU device is selected in Setup step). 

When prompted with `Enter BERTweet model path:`, enter the file path to the location of `mtl_bertweet.pt` (e.g. `model/mtl_bertweet.pt`)

After the execution of `mtl/bertweet.pyc` is complete, you will find the output file at `dataset/output/mtl/bertweet_output.csv`

## Running COMET-ATOMIC Models

Run `python comet-atomic/BART.pyc` 

When promped with `Enter GPU device name:`, enter either `cuda:0` or `cuda:1` (based on which GPU device is selected in Setup step). 

When prompted with `Enter BART model path:`, enter the file path to the location of `comet-atomic_2020_BART` (e.g. `model/comet-atomic_2020_BART`)

After the execution of `comet-atomic/BART.pyc` is complete, you will find the output file at `dataset/output/comet-atomic/BART_output.csv`









