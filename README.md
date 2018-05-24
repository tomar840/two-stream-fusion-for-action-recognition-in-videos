# two-stream-fusion-for-action-recognition-in-videos
We have implemented convolutional two stream network for action recognition for two cases - 
  * Two stream average fusion at softmax layer.
  * Two stream fusion at convolutional layer.
## 1. Data
We have used UCF101 dataset for this project.
For utilization of temporal information, we have used optical flow images and RGB frames for utilizing spatial information.
The pre-processed RGB frames and flow images can be downloaded from [feichtenhofer/twostreamfusion](https://github.com/feichtenhofer/twostreamfusion))
  * RGB images
  ```
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.001
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.002
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.003
  
  cat ucf101_jpegs_256.zip* > ucf101_jpegs_256.zip
  unzip ucf101_jpegs_256.zip
  ```
  * Optical Flow
  ```
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.001
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.002
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.003
  
  cat ucf101_tvl1_flow.zip* > ucf101_tvl1_flow.zip
  unzip ucf101_tvl1_flow.zip
  ```
For both the cases we have used a stack of 10 RGB frames as input for spatial stream and a stack of 50 optical flow frames as input for temporal stream.
So, for a batch size = 4, typical spatial loader will look like the image below - 
![data_loading jpeg](https://user-images.githubusercontent.com/37335834/40482183-7ea4b7f6-5f71-11e8-93c0-f0867537e107.jpeg)

## 2. Models
We have used vgg-19 model pre-trained on ImageNet for both the streams. 

## 3. Implementation details for both cases
  * Note :- To do weight transformation for first layers of ConvNets, we first average the weight value across the RGB channels and replicate this average value by the channel number in that ConvNet.
  ### 3.1 Two stream average fusion at softmax layer
    The architecture for this case has been shown in the figure below - 
![average_fusion](https://user-images.githubusercontent.com/37335834/40482038-f9e92a4c-5f70-11e8-8abe-db06eee3feb9.jpeg)
   

  ### 3.2 Two stream fusion at convolution layer
  The Architecture for this case is shown in the Figure below - 
![conv_fusion](https://user-images.githubusercontent.com/37335834/40482277-e704793a-5f71-11e8-8ae0-dc6b32bb58d9.jpeg)
  The ConvNets are being replaced be vgg model, trained on ImageNet. 
  
  ## 4. Training Models
  * Please modify this [path](https://github.com/tomar840/two-stream-average-fusion-for-action-recognition-in-videos/blob/master/average_fusion.py#L213) and this [path](https://github.com/tomar840/two-stream-average-fusion-for-action-recognition-in-videos/blob/master/conv_fusion.py#L206) to fit the UCF101 dataset on your device.
  * If you want to change the number of frames in RGB stack, then modify [here](https://github.com/tomar840/two-stream-average-fusion-for-action-recognition-in-videos/blob/master/dataloader/spatiotemporal_loader.py#L83). Select the frames you want to have in the stack. If you want, you can also introduce randomness in choosing the frames for stacking. 

## 5. Performance
 ### 5.1 Performance for two stream average fusion 
 * For first 20 classes of UCF101 dataset

 Network      | Acc.  |
--------------|:-----:|
Spatial cnn   | 82.1% | 
Motion cnn    | 79.4% | 
Average fusion| 88.5% |

 * For all 101 classes of UCF101 dataset

 Network      | Acc.  |
--------------|:-----:|
Spatial cnn   | 82.1% | 
Motion cnn    | 79.4% | 
Average fusion| 88.5% |

 ### 5.2 Performance for two stream fusion at convolution layer
 * For first 20 classes of UCF101 dataset, we get an accuracy of 96.01 % 
 * For all 101 classes of UCF 101 dataset, we get an accuracy of 68.23 % 

## 6. Reference Paper
*  [[1] Two-stream convolutional networks for action recognition in videos](http://papers.nips.cc/paper/5353-two-stream-convolutional)
*  [[2] Convolutional Two-Stream Network Fusion for Video Action Recognition](https://arxiv.org/pdf/1604.06573.pdf)


