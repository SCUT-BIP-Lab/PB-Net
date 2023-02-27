# PB-Net: physiological-behavioral characteristic understanding network
Pytorch Implementation of paper:

> **Understanding Physiological and Behavioral Characteristics Separately for High-Performance Dynamic Hand Gesture Authentication**
>
> Wenwei Song, Wenxiong Kang\*, and Yufeng Zhang.

## Main Contribution
Dynamic hand gesture authentication is a challenging fine-grained video understanding task. It requires models to have good spatiotemporal analysis capabilities so as to extract stable physiological and behavioral features from gesture videos for identity verification.  Inspired by biological studies, we present a physiological-behavioral characteristic understanding network (PB-Net) for hand gesture authentication. According to the properties of physiological and behavioral characteristics, we design two branches, physiological branch (P-Branch) and behavioral branch (B-Branch), as well as corresponding data tailoring strategies for the PB-Net. The data tailoring strategies can produce two customized videos for the two branches to facilitate the analyses of physiological and behavioral characteristics and remove significant redundant information to improve running efficiency. The P-Branch and B-Branch do not interfere with each other and focus on the distillation of physiological and behavioral features separately. Considering that the importance degree of the physiological and behavioral features could be different and changeable, we devise an adaptive physiological-behavioral feature fusion (APBF) module to automatically assign appropriate weights for the two features and merge them together to obtain an ideal identity feature. Finally, the rationality, validity, and superiority of the PBNet are fully demonstrated by extensive ablation experiments and sufficient comparisons with 23 excellent video understanding networks on the SCUT-DHGA dataset.
 <div align="center">
 <p align="center">
  <img src="https://raw.githubusercontent.com/SCUT-BIP-Lab/PB-Net/master/img/PBNet.png" />
 </p>
</div>

 The overall architecture of the PB-Net.  It contains four core components: data tailoring module, P-Branch, B-Branch, and APBF module. The data tailoring module is responsible for processing the input video according to the natures of physiological and behavioral characteristics, making the derived videos more suitable for the P-Branch and B-Branch to perform the analysis of physiological and behavioral characteristics, respectively, while removing redundant information to improve running efficiency. The backbones of both P-Branch and B-Branch are pre-trained ResNet18, using the TSN paradigm for video processing. Since each frame of the B-Branch's input video comprises three grayscale frames derived from the original video, the B-Branch can perform temporal analysis by channel information aggregation. Besides, we also insert a temporal convolution with a residual connection (TempConv) in each of the Conv1 (C1) and Layer1(L1) to further explore the behavioral characteristics. After feature extraction, the APBF module will adaptively assign weights to the physiological and behavioral features from the two branches and output the final identity feature via concatenation. For the network training, we use the AMSoftmax loss function and train the two branches separately using two independent optimizers. To obtain valuable fusion weights, we employ a third optimizer focusing on the training of the APBF module with a preference to assign a greater weight for the behavioral feature. The "scissor" indicate that the gradients from the APBF are not passed to the two branches during backpropagation. The GSAP and GTAP denote global spatial average pooling and global temporal average pooling, respectively.


## Comparisons with SOTAs
To prove the rationality and superiority of our PB-Net, we conduct extensive experiments on the SCUT-DHGA dataset. The EERs shown in the figure are all average values over six test configurations on the cross session.

 <div align="center">
 <p align="center">
  <img src="https://raw.githubusercontent.com/SCUT-BIP-Lab/PB-Net/master/img/efficiency statistics.png" />
 </p>
</div>

  EER vs. FLOPs trade-off under MG evaluation protocol. We select 23 representative video understanding networks from the areas of action recognition, hand gesture authentication, and gait recognition, covering 3D CNN, 2D CNN, Symbiotic CNN, and Two-stream CNN (TS CNN). The networks with "*" indicate SOTA hand gesture authentication networks. The numbers next to the networks are FLOPs. It is very obvious that our PB-Net has significant advantages in both EER and FLOPs.

## Dependencies
Please make sure the following libraries are installed successfully:
- [PyTorch](https://pytorch.org/) >= 1.7.0

## How to use
This repository is a demo of PB-Net. Through debugging ([main.py](/main.py)), you can quickly understand the 
configuration and building method of [PB-Net](/model/PBNet.py).

If you want to explore the entire dynamic hand gesture authentication framework, please refer to our pervious work [SCUT-DHGA](https://github.com/SCUT-BIP-Lab/SCUT-DHGA) 
or send an email to Prof. Kang (auwxkang@scut.edu.cn).
