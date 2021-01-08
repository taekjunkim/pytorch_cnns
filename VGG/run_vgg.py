##### model part
import torch
import torch.nn

class VGG(nn.Module):
    def __init__(self):
        super().__init__();
        ### feature
        self.feature = nn.Sequential(
                           # nn.Conv2d(input_size,output_size,kernel_size,stride,padding);
                           #     padding: zero-padding on both sides;          
                           #     input: 3x224x224;    output: 64x224x224 <== (224+1+1-3)/1 + 1
                           nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                           # nn.ReLu();
                           #     input: 64x224x224;    output: 64x224x224
                           nn.ReLU(),
                           # nn.Conv2d(input_size,output_size,kernel_size,stride,padding);
                           #     padding: zero-padding on both sides;          
                           #     input: 64x224x224;    output: 64x224x224 <== (224+1+1-3)/1 + 1
                           nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                           # nn.ReLu();
                           #     input: 64x224x224;    output: 64x224x224
                           nn.ReLU(),
                           # nn.MaxPool2d();
                           #     input: 64x224x224;    output: 64x112x112 <== (224-2)/2 + 1
                           nn.MaxPool2d(kernel_size=2, stride=2), 
                           #     input: 64x112x112;    output: 128x112x112 <== (112+1+1-3)/1 + 1
                           nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                           nn.ReLU(),
                           #     input: 128x112x112;    output: 128x112x112 <== (112+1+1-3)/1 + 1
                           nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                           nn.ReLU(),
                           #     input: 128x112x112;    output: 128x56x56 <== (112-2)/2 + 1
                           nn.MaxPool2d(kernel_size=2, stride=2), 
                           #     input: 128x56x56;    output: 256x56x56 <== (56+1+1-3)/1 + 1
                           nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                           nn.ReLU(),
                           #     input: 256x56x56;    output: 256x56x56 <== (56+1+1-3)/1 + 1
                           nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                           nn.ReLU(),
                           #     input: 256x56x56;    output: 256x56x56 <== (56+1+1-3)/1 + 1
                           nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                           nn.ReLU(),
                           #     input: 256x56x56;    output: 256x28x28 <== (56-2)/2 + 1
                           nn.MaxPool2d(kernel_size=2, stride=2), 
                           #     input: 256x28x28;    output: 512x28x28 <== (28+1+1-3)/1 + 1
                           nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                           nn.ReLU(),
                           #     input: 512x28x28;    output: 512x28x28 <== (28+1+1-3)/1 + 1
                           nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                           nn.ReLU(),
                           #     input: 512x28x28;    output: 512x28x28 <== (28+1+1-3)/1 + 1
                           nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                           nn.ReLU(),
                           #     input: 512x28x28;    output: 512x14x14 <== (28-2)/2 + 1
                           nn.MaxPool2d(kernel_size=2, stride=2), 
                           #     input: 512x14x14;    output: 512x14x14 <== (14+1+1-3)/1 + 1
                           nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                           nn.ReLU(),
                           #     input: 512x14x14;    output: 512x14x14 <== (14+1+1-3)/1 + 1
                           nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                           nn.ReLU(),
                           #     input: 512x14x14;    output: 512x14x14 <== (14+1+1-3)/1 + 1
                           nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                           nn.ReLU(),
                           #     input: 512x14x14;    output: 512x7x7 <== (14-2)/2 + 1
                           nn.MaxPool2d(kernel_size=2, stride=2), 
                           );
        ### classifier
        self.classifier = nn.Sequantial(
                              nn.Linear(512 * 7 * 7, 4096),
                              nn.ReLU(inplace=True),
                              nn.Dropout(),
                              nn.Linear(4096, 4096),
                              nn.ReLU(inplace=True),
                              nn.Dropout(),
                              nn.Linear(4096, 1000)                              
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
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=10, shuffle=True, num_workers=2);


##### training part
# Construct a loss function and an Optimizer. 
loss_fun = torch.nn.MSELoss(reduction='sum');
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9);

VGG.train();
for epoch in range(100):
    for i, (images,labels) in enumerate(dataloaders):    
        input = images;
        y_act = labels;
        y_pred = VGG(input);
        loss = loss_fun(y_pred,y_act);

        # Zero gradients before a backward pass, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
