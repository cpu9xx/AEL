# Adaptive-Ensemble-Learning
An Ensemble View on Denoising Recommendation.

# Overview
AEL contains three modules: corrupt module, adaptive ensemble module, and denoising module. In the denoising module, we first construct three sub-AEs as components based on the Collaborative Denoising Autoencoder~\cite{CDAE}. Then, we vary the denoising capacities of three parent-AEs and significantly reduce their model size using a novel method. This method first creates three sub-AEs as components, then stacks them to construct heterogeneous parent-AEs. We also introduce a corrupt module to improve robustness by partially corrupting initial input, preventing sub-AEs from simply learning the identity function. The adaptive ensemble module achieves the denoising capacity adaptability. It contains an improved sparse gating network as a brain, which can analyze the historical performance of parent-AEs, and automatically select the two most suitable parent-AEs to synthesize appropriate denoising capacity for current input data.

# Requirements
The model is implemented using PyTorch. The versions of packages used are shown below.
- torch==1.13.0
- CUDA==11.6
- numpy==1.24.4
- pandas==1.4.2
- matplotlib==3.7.1
- easydict==1.13

# Special thanks 
Very thanks to Dr.Wenjie Wang with his code [DenoisingRec](https://github.com/WenjieWWJ/DenoisingRec).

# Run
- Step 1. Train parent-AEs, or contact us to get pre-trained models weipuchenn@gmail.com: python train.py
- Step 2. Train sparse gating network: python moe_train.py
