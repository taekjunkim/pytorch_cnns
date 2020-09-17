import torch
import torch.nn

## model part
class AlexNet(nn.Module):
    def __init__(self, init_weights=True):
        super().__init__();
        # feature
        self.feature = nn.Sequential(OrderedDict([
                           # nn.Conv2d(input_size,output_size,kernel_size,stride,padding);
                           #     padding: zero-padding on both sides;          
                           #     input: 3x224x224;    output: 64x55x55 <== (224+2+2-11)/4 + 1
                           ('conv0', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                           #     input: 64x55x55;    output: 64x55x55
                           ('relu0', nn.ReLU()),
                           #     input: 64x55x55;    output: 64x27x27 <== (55-3)/2 + 1
                           ('maxpool0', nn.MaxPool2d(kernel_size=3, stride=2)), 
                            
                            
                           ]));
        

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),        
        # classifier
        
        
        
        if init_weights:
            self._initialize_weights();
            
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)            
