# two-stream-average-fusion-for-action-recognition-in-videos
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
  
## 2. Models
We have used vgg-19 model pre-trained on ImageNet for both the streams. 

## 3. Implementation details for both cases
For both the cases we have used a stack of 10 RGB frames as input for spatial stream and a stack of 50 optical flow frames as input for temporal stream.
  * Note :- To do weight transformation for first layers of ConvNets, we first average the weight value across the RGB channels and replicate this average value by the channel number in that ConvNet.
  ### 3.1 Two stream average fusion at softmax layer
    The architecture for this case has been shown in the figure below - 
