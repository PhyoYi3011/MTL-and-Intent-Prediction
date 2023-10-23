# Unveiling the Dynamics of Crisis Events: Sentiment and Emotion Analysis via Multi-Task Learning with Attention Mechanism and Subject-based Intent Prediction

## Multi-Task Learning Model

Fine-tuned MTL models (BERT, RoBERTa, BERT) can be downloaded [here](https://drive.google.com/drive/folders/1xNaOG-4VS2emW2N7IStLm-E2EzAltbde?usp=sharing).

## COMET-ATOMIC 2020 Model

COMET-ATOMIC 2020 models (BART, GPT2-XL) can be downloaded [here](https://github.com/allenai/comet-atomic-2020).


## Prerequisites to Run Models

* NVIDIA CUDA toolkit. Refer to this installation [guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) for Linux or [guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) for Microsoft Windows.
* NVIDIA container toolkit. Refer to this installation [guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
* To find out if pytorch is using GPU, run below python script.
  * import torch
  * torch.cuda.is_available()
  * torch.cuda.current_device()
   * “0” if “cuda:0” is used.
   * “1” if “cuda:1” is used. 
