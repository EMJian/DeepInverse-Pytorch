'''
Created on Apr 10, 2019

@author: chengzi
'''

import os,sys,glob,math
from PIL import Image
import numpy as np
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from DepInvercs_model import DeepInverse

  
block_size =33;
dtype = torch.float32


def createDir(imgn,dirname,CS_ratio):
    img_path = os.path.dirname(imgn)        
    img_path = os.path.abspath(os.path.join(img_path, "..")) + dirname
    img_rec_path = "%s_rec_%s" % (img_path,CS_ratio)
    isExists=os.path.exists(img_rec_path)
    if not isExists:
        os.makedirs(img_rec_path)
        return img_rec_path
    else:
        return img_rec_path
    
def psnrISTA(img1, img2):
    img1=img1.astype(np.float32)
    img2=img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def psnr(recovered, original):
    recovered=recovered.astype(np.float32)
    original=original.astype(np.float32)    
    recovered = torch.from_numpy(recovered)
    original = torch.from_numpy(original)
    
    mse = F.mse_loss(recovered, original)
    if mse == 0:
        return 100
    psnr = 10 * np.log10(1 / mse.item())
    return psnr


def RGBrec(model,csinput,device,img_orig, channels_Num, row_new, col_new):

#     [row, col,channels_Num] = img_orig.shape 
    row = img_orig.shape[0]
    col = img_orig.shape[1]
         
    with torch.no_grad():
        csinput = csinput.to(device=device, dtype=dtype)
        img_recch = model(csinput)
    print(img_recch.shape)   
    
    # Use Tensor.cpu() to copy the tensor to host memory first
    img_recch = img_recch.cpu().numpy()

    row_block = int(row_new/block_size)
    col_block = int(col_new/block_size)
    blocknum = int(row_block*col_block) 

    RGB_rec = []    
    rec_PSNR =0.0  
    img_x = np.zeros([row_new, col_new], dtype=np.float32)
    
    for channel_no in range(channels_Num):
        begblockid = blocknum*channel_no  
        endblockid = blocknum*(channel_no+1)             
        img_rec = img_recch[begblockid:endblockid,:,:,:]
        
        count = 0
        for xi in range(row_block):
            for yj in range(col_block):          
                img_x[xi*block_size:(xi+1)*block_size, yj*block_size:(yj+1)*block_size] = img_rec[count,:,:,:]  
                count = count +1
        imgarrf_x = img_x[:row, :col]
        imgf_x = Image.fromarray(np.clip(imgarrf_x * 255, 0, 255).astype(np.uint8))
#         imgf_x.show()
        RGB_rec.append(imgf_x) 
        
        if channels_Num==3:
            rec_PSNR = rec_PSNR + psnr(imgarrf_x, img_orig[:,:,channel_no])
    #         rec_PSNR = rec_PSNR + psnrISTA(imgarrf_x*255, img_orig[:,:,channel_no]*255)
        else:
            rec_PSNR = rec_PSNR + psnr(imgarrf_x, img_orig)
    
    if channels_Num==3:
            RGBimg_rec=Image.merge("RGB", (RGB_rec[0],RGB_rec[1],RGB_rec[2]))
            rec_PSNR = rec_PSNR /3.0   
    elif channels_Num==1:
            RGBimg_rec=RGB_rec[0]
            rec_PSNR = rec_PSNR            
#     RGBimg_rec.show()
           
    return RGBimg_rec,rec_PSNR
    
    
def PRWimgTensor(imgpath,phi):
    
    img_rgb = Image.open(imgpath);
#   plt.show(img_rgb)
    img = np.array(img_rgb, dtype=np.uint8)   
    img_bsize = sys.getsizeof(img);  
    
#     [row, col, channels_Num] = img.shape
    channels_Num = len(img_rgb.split())
    row = img.shape[0]
    col = img.shape[1]
    
    if np.mod(row,block_size)==0:
        row_pad=0
    else:    
        row_pad = block_size-np.mod(row,block_size)
    
    if np.mod(col,block_size)==0:
        col_pad = 0
    else:        
        col_pad = block_size-np.mod(col,block_size)
    row_new = row + row_pad
    col_new = col + col_pad
    row_block = int(row_new/block_size)
    col_block = int(col_new/block_size)
    blocknum = int(row_block*col_block)    
    
    img_ycs = []
    ysize = 0.0            
    for channel_no in range(channels_Num):
#         print("channel no ====%d"%(channel_no))
        if channels_Num==1:        
            imgorg=img[:,:]
        else:
            imgorg=img[:,:,channel_no]
        Ipadc = np.concatenate((imgorg, np.zeros([row, col_pad],dtype=np.uint8)), axis=1)
        Ipadc = np.concatenate((Ipadc, np.zeros([row_pad, col+col_pad],dtype=np.uint8)), axis=0) 
        Ipadc = Ipadc/255.0   
#         [row_new, col_new] = Ipadc.shape

        img_x = np.zeros([blocknum, 1,block_size, block_size], dtype=np.float32)
        count = 0
        for xi in range(row_block):
            for yj in range(col_block):            
                img_x[count] = Ipadc[xi*block_size:(xi+1)*block_size, yj*block_size:(yj+1)*block_size]  
                count = count +1
        img_x = torch.from_numpy(img_x)
        
        X = torch.empty(blocknum, 1, block_size, block_size, dtype=torch.float)   
        for i in range(X.shape[0]):         
            y = torch.mv(phi, img_x[i].view(-1) )   # Performs a matrix-vector product
            # You cannot use sys.getsizeof(y) to get the correct memory size of the tensor y
            # https://stackoverflow.com/questions/54361763/pytorch-why-is-the-memory-occupied-by-the-tensor-variable-so-small
            ysize = ysize + sys.getsizeof(y.storage());
            x_tilde = torch.mv(phi.transpose(0,1), y)            
            x_tilde = x_tilde.view(1, 1, block_size, block_size)  # view as 1-channel 32x32 image
            X[i] = x_tilde
        img_ycs.append(X)   
    img_ycs22 = torch.cat(img_ycs)
    
    return img/255.0, channels_Num, img_bsize, ysize, img_ycs22, row_new, col_new

def DeepInvertCS(filepaths,fname_phi,fname_sdict, CS_ratio):
        # set up device
    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('using device:', device)
     
#   To load the model, use the following code:
    with open(fname_phi, 'rb') as f:
        phi = pickle.load(f)
    model = DeepInverse(phi.shape)
    model.load_state_dict(torch.load(fname_sdict))
    model.eval()
    model.to(device)
    print('trained model loaded')    

    ImgNum = len(filepaths)
    PSNR_All = np.zeros([ImgNum], dtype=np.float32) 
    MCRy = np.zeros([ImgNum], dtype=np.float32) 
    img_rec_path = createDir(filepaths[0],'/DVCS',(CS_ratio)[2:4]);
    
    for img_no in range(ImgNum): 
        imgName = filepaths[img_no]    
        img_orig, channels_Num, img_bsize, ysize, img_ycs, row_new, col_new= PRWimgTensor(imgName,phi) 
        MCRy[img_no] = img_bsize/ysize
        
        RGBimg_rec,rec_PSNR =  RGBrec(model, img_ycs, device, img_orig, channels_Num, row_new, col_new) 
        PSNR_All[img_no] = rec_PSNR
    
        print("Image %s, PSNR= %.6f, mCR= %0.3f" % (imgName, rec_PSNR, MCRy[img_no]))
        
        img_name = os.path.split(imgName)[-1]        
        img_rec_name = "%s/%s" % (img_rec_path, img_name)    
        RGBimg_rec.save(img_rec_name) 
        print("Rec_image save to",img_rec_name) 
    #-------------------------------------------------
    print("-----------------------")     
    output_data = "CS_ratio= %.2f , AvgPSNR is %.2f dB, mCR is %.3f \n" % (float(CS_ratio), np.mean(PSNR_All), np.mean(MCRy))
    print(output_data)    
    
#     plt.subplot(1,2,1)
#     plt.imshow(img_orig); plt.title('original')
#     plt.subplot(1,2,2)
#     plt.imshow(RGBimg_rec); plt.title('restored')
#     plt.show()
    
if __name__ == '__main__':   

    path_dataset = "/home/chengzi/Desktop/workspace20170624/DeepInvertCS/Test_Image"
    filepaths = glob.glob(path_dataset + '/*.tif')
    
    path_dataset = "/media/chengzi/FT-dataset/PRW-v16.04.20/testprwdemo"    
    filepaths = glob.glob(path_dataset + '/*.jpg')
    
    
    csrate = '0.01'
    
    fname_sdict = "dvcs_91imgcs_%s_gray.pt" % (csrate)[2:4]
    fname_phi = "dvcs_91imgcs_%s_gray-measurement.pickle" % (csrate)[2:4]
    
    DeepInvertCS(filepaths,fname_phi,fname_sdict,csrate)


