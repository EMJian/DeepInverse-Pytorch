# DeepInverse Re-implementation
Re-implements the Re-implement the Compressive Sensing (CS) Network DeepInverse described in Learning to invert: Signal recovery via Deep Convolutional Networks. (https://ieeexplore.ieee.org/abstract/document/7952561, https://arxiv.org/pdf/1701.03891.pdf )

## The implementation mainly uses pytorch 0.4.1. 

## Training data (T91 dataset)   
Downloaded at （http://vllab.ucmerced.edu/wlai24/LapSRN/ ）or (https://drive.google.com/open?id=1AoEcNA5-onnSqBcWZawNw7ZFrJ1fFR_C)

# Results on the test datast Set11  

 MR=0.01    PSNR=16.74   
 MR=0.04    PSNR=18.33   
 MR=0.10    PSNR=20.38  
 MR=0.25    PSNR=20.90    
 MR=0.40    PSNR=22.23  
 MR=0.50    PSNR=23.19  

## Refs:
 - DeepInverse in Pytorch （https://github.com/y0umu/DeepInverse-Reimplementation )  
 - ISTA-Net in Tensorflow （https://github.com/jianzhangcs/ISTA-Net)  
 - CSNet in MatconvNet（https://github.com/wzhshi/CSNet)  
 - ReconNet in matCaffe (https://github.com/AtenaKid/Caffe-DCS) and (https://github.com/KuldeepKulkarni/ReconNet) 
 - Adaptive ReconNet in Tensorflow (https://github.com/yucicheung/AdaptiveReconNet)  
 - Reproducible deep learning for compressive sensing (https://github.com/AtenaKid/Reproducible-Deep-Compressive-Sensing)  
 
 - D-AMP (https://github.com/ricedsp/D-AMP_Toolbox)
 - IRCNN (https://github.com/cszn/IRCNN), (https://github.com/lipengFu/IRCNN) and (https://github.com/YunzeMan/IRCNN)
 - DnCNN （https://github.com/cszn/DnCNN ）
