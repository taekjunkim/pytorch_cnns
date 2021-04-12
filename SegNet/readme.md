I implemented a SegNet based on VGG16 which does semantic pixel-wise segmentation. 
The SegNet consists of five layers of encoders and five layers of decoders.
Parameters in encoders were freezed during the training, so that their weights were the same as those in VGG16. 

SegNet was trained on VOC2012 dataset 
- training set: 2914 images
- test set: 210 images

Because of the small size of training set, the model performance is not great. 
But we can understand how the model is created and being trained. 
