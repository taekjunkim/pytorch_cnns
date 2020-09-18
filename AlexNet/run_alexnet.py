##### model part
import torch
import torch.nn

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__();
        ### feature
        self.feature = nn.Sequential(
                           # nn.Conv2d(input_size,output_size,kernel_size,stride,padding);
                           #     padding: zero-padding on both sides;          
                           #     input: 3x224x224;    output: 64x55x55 <== (224+2+2-11)/4 + 1
                           nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                           # nn.ReLu();
                           #     input: 64x55x55;    output: 64x55x55
                           nn.ReLU(),
                           # nn.MaxPool2d();
                           #     input: 64x55x55;    output: 64x27x27 <== (55-3)/2 + 1
                           nn.MaxPool2d(kernel_size=3, stride=2), 
                           #     input: 64x27x27;    output: 192x27x27 <== (27+2+2-5)/1 + 1
                           nn.Conv2d(64, 192, kernel_size=5, padding=2),
                           nn.ReLU(),
                           #     input: 192x27x27;    output: 192x13x13 <== (27-3)/2 + 1
                           nn.MaxPool2d(kernel_size=3, stride=2), 
                           #     input: 192x13x13;    output: 384x13x13 <== (13+1+1-3)/1 + 1
                           nn.Conv2d(192, 384, kernel_size=3, padding=1),
                           nn.ReLU(),
                           #     input: 384x13x13;    output: 256x13x13 <== (13+1+1-3)/1 + 1
                           nn.Conv2d(384, 256, kernel_size=3, padding=1),
                           nn.ReLU(),
                           #     input: 256x13x13;    output: 256x13x13 <== (13+1+1-3)/1 + 1
                           nn.Conv2d(384, 256, kernel_size=3, padding=1),
                           nn.ReLU(),
                           #     input: 256x13x13;    output: 256x6x6 <== (13-3)/2 + 1
                           nn.MaxPool2d(kernel_size=3, stride=2) 
                           );
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6)); ### I think it may not be needed, if the input size is already 256x6x6                        
        ### classifier
        self.classifier = nn.Sequantial(
                              nn.Dropout(),
                              nn.Linear(256 * 6 * 6, 4096),
                              nn.ReLU(inplace=True),
                              nn.Dropout(),
                              nn.Linear(4096, 4096),
                              nn.ReLU(inplace=True),
                              nn.Linear(4096, num_classes)                              
                              );  
        ### initialize weights
        self._initialize_weights();
            
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu');
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0);
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1);
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01);
                nn.init.constant_(m.bias, 0);            

    def forward(self,x):
        x = self.features(x);
        x = self.avgpool(x);
        x = torch.flatten(x, 1);
        x = self.classifier(x);
        return x;        
    
##### data loading part
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Comment 0: define transformation that you wish to apply on image
data_transforms = transforms.Compose(
    #Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] 
    #to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] 
    #if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    # or if the numpy.ndarray has dtype = np.uint8
    [transforms.ToTensor(),
     #normalization of pre-trained networks:
     #from docs:
     #All pre-trained models expect input images normalized in the same way, 
     #i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where
     #H and W are expected to be at least 224. The images have to be loaded 
     #in to a range of [0, 1] and then normalized using 
     #mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. 
     transforms.Normalize([0.485, 0.456, 0.406], 
                          [0.229, 0.224, 0.225])])#

# Comment 1 : Load the datasets with ImageFolder
image_datasets = datasets.ImageFolder(root= stim_path, transform=data_transforms)
# Comment 2: Using the image datasets and the transforms, define the dataloaders
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=10, shuffle=False, num_workers=2);


##### training part
