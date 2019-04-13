'''
https://github.com/y0umu/DeepInverse-Reimplementation
'''
# imports
import glob
import scipy.io as sio
from PIL import Image
import numpy as np
from six.moves import cPickle as pickle
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter 
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import sampler
import torchvision.transforms as T
# --------------
from DepInvercs_model import DeepInverse


#####################################################
# set up device
USE_GPU = True
dtype = torch.float32 # we will be using float throughout this tutorial
block_size =32;
block_size =33;

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('using device:', device)
# Constant to control how frequently we print train loss
print_every = 100

def getPhi(csrate):
    N = 32*32
    M = int(N * csrate)
    phi = torch.randn(M, N)
    return phi

def loadPhi(csrate):    
    Phi_data_Name = 'phi_0_%s_1089.mat' % csrate[2:4]
    Phi_data = sio.loadmat(Phi_data_Name)
#     Phi_input = Phi_data['phi'].transpose()
    Phi_input = Phi_data['phi']
    Phi_input = Phi_input.astype(np.float32)
    phi = torch.from_numpy(Phi_input)
        
    return phi
    
# writing custom Datasets + Samplers
# https://www.pytorchtutorial.com/pytorch-custom-dataset-examples/
# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
class load91imgDataset(Dataset):
    def _PRWGrayImgTensor(self, imgpath): 
        img_rgb = Image.open(imgpath).convert('L');
        plt.show(img_rgb)        
        img = np.array(img_rgb, dtype=np.uint8)
    #     channels_Num = len(img_rgb.split())
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
        
        row_new = row - row_pad
        col_new = col - col_pad
        row_block = int(row_new/block_size)
        col_block = int(col_new/block_size)
        blocknum = int(row_block*col_block)    
        
        imgorg=img[0:row_new,0:col_new]   
        Ipadc = imgorg/255.0   
    
        img_x = np.zeros([blocknum, 1,block_size, block_size], dtype=np.float32)
        count = 0
        for xi in range(row_block):
            for yj in range(col_block):            
                img_x[count] = Ipadc[xi*block_size:(xi+1)*block_size, yj*block_size:(yj+1)*block_size]  
                count = count +1
        img_x = torch.from_numpy(img_x)
        
        X = torch.empty(blocknum, 1, block_size, block_size, dtype=torch.float)   
        for i in range(X.shape[0]):  
            X[i] = self.transform(img_x[i])     
        
        return X
    
    
    'Characterizes a dataset for PyTorch'
    def __init__(self, filepaths, phi):
        'Initialization'
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Grayscale(),
                T.ToTensor(),
                T.Normalize([0.5], [0.5])
            ])
        self.phi = phi  
        
        ImgNum = len(filepaths)
        imgblks = []
        for img_no in range(ImgNum): 
            imgpath = filepaths[img_no]
            imgblock = self._PRWGrayImgTensor(imgpath) 
            imgblks.append(imgblock) 
                  
        self.images = torch.cat(imgblks,0)
        print("----", imgblock.shape, '==',self.images.shape,'  \n')

        # measurements = []
        x_tildes = []
        for im in self.images:
            y = torch.mv(self.phi, im.view(-1) )   # Performs a matrix-vector product
            x_tilde = torch.mv( self.phi.transpose(0,1), y)
            x_tilde = x_tilde.view(1, 1, block_size, block_size)  # view as 1-channel 32x32 image
            x_tildes.append( x_tilde )
        # self.measurements = torch.cat(measurements)
        self.x_tildes = torch.cat(x_tildes)
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.images)

    def __getitem__(self, index):
        'Generates one sample of data'
        return (self.x_tildes[index], self.images[index])
    
    
def train91Img(model, optimizer, loader_train, epochs=1, logdir=None):
    """
    Train a model on 91images using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    - logdir: string. Used to specific the logdir of tensorboard
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    writer = SummaryWriter(log_dir=logdir)
    print("Run `tensorboard --logdir={logdir} --host=127.0.0.1` to visualize in realtime")
    loss_history = []
    tfx_steps = 0
    for e in range(epochs):
        print('-----------------------------')
        print('* epoch {e+1}/{epochs}')
        for t, (measurement, original_im) in enumerate(loader_train):  
            model.train()  # put model to training mode
            measurement = measurement.to(device=device, dtype=dtype)
            original_im = original_im.to(device=device, dtype=dtype)  # move to device, e.g. GPU

            recovered_im = model(measurement)
#             ipdb.set_trace()
            loss_fn = nn.MSELoss()
            loss = loss_fn(recovered_im, original_im)

            # Zero out all of the gradients for the variables which the optimizer will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()
            
            loss_history.append(loss.item())
            writer.add_scalar('train/loss', loss.item(), tfx_steps)
            
            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
            tfx_steps += 1
    
    # plot everything after the loop is over
    writer.close()  # tensorboardX writer
    plt.plot(loss_history, 'o'); plt.title('Training loss'); plt.xlabel('Iteration')
    plt.show()

if __name__ == '__main__':  
    
    datapath = "/home/chengzi/Desktop/workspace20170624/DeepInvertCS/Train91img"
    filepaths = glob.glob(datapath + '/*.bmp')
    
    csrate = '0.01'    
#     phi = getPhi(csrate)
    phi = loadPhi(csrate)
    
    # load dataset
    train_set = load91imgDataset(filepaths, phi)
    NUM_TRAIN = int(0.8 * len(train_set) )
    TOTAL_SAMPLES = len(train_set)
    loader_train = DataLoader(train_set, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))    
    print("Dataset loaded")
##################################################### 

 # train!
    exp_name = 'exp15'
    model = DeepInverse(phi.shape)
    learning_rate = 5e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train91Img(model, optimizer,loader_train, epochs=5, logdir='runs/' + exp_name + '_1')
    print("Stage 1 of train is done. \n")    
    
    learning_rate = 1e-5
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train91Img(model, optimizer, loader_train, epochs=5, logdir='runs/' + exp_name + '_2')
    print("Stage 2 of train is done. \n")

    # Save the state_dict of the model
    fname_sdict = 'dvcs-img91_gray.pt'
    fname_phi = 'dvcs-img91_gray-measurement.pickle'
    
    fname_sdict = "dvcs_91imgcs_%s_gray.pt" % (csrate)[2:4]
    fname_phi = "dvcs_91imgcs_%s_gray-measurement.pickle" % (csrate)[2:4]
    
    torch.save(model.state_dict(), fname_sdict)
    with open(fname_phi, 'wb') as f:
        pickle.dump(phi, f)
    
    print("Trained model is saved. \n")

