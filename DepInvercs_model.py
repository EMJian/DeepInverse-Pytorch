'''
https://github.com/y0umu/DeepInverse-Reimplementation
'''
import torch.nn as nn

# The model
class DeepInverse(nn.Module):
    def _init_weights(self, m):
        print(m)
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)
#         if type(m) == nn.Linear:
#             nn.init.kaiming_normal_(m.weight.data)
#             nn.init.constant_(m.bias.data, 0)
    
    def __init__(self, mmat_shape):
        '''
        - mmat_shape: the shape of the measurement matrix
        '''
        super().__init__()
#         self.mmat_shape = mmat_shape
#         self.linear = nn.Linear(mmat_shape[0], mmat_shape[1], bias=False)
        self.conv_bundle = nn.Sequential(
            nn.Conv2d(1, 64, 11, stride=1, padding=5, dilation=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            nn.Conv2d(64, 32, 11, stride=1, padding=5, dilation=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            nn.Conv2d(32, 1, 11, stride=1, padding=5, dilation=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU()
        )
#         self.linear.apply(self._init_weights)
        self.conv_bundle.apply(self._init_weights)
        
        
    def forward(self, x):
#         ipdb.set_trace()
#         x = self.linear(x)
#         num_samples = x.shape[0]
#         x = x.view(num_samples, 3, 32, 32)  # view as 3-channel 32x32 image
        x = self.conv_bundle(x)  # should looks like some original image?        
        return x
    
    
    