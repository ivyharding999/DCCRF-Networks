# Discrete Convolutional CRF Networks for Depth Estimation from Monocular Infrared Images
This repository contains the reference implementation for our proposed DCCRF networks in Temsorflow.

This paper focuses on depth estimation from monocular infrared images, which is essential for understanding the structure of 3D scene and can promote the development of night vision applications. 

The infrared images,corresponding ground-truth depths and depth estimation results(by DCCRF) are shown below:
  
  ![image](https://github.com/ivyharding999/Discrete-Convolutional-CRF-Networks-for-Depth-Estimation-from-Monocular-Infrared-Images/blob/master/Infrared%20images/data.png)

 # Requirements
 
 ## Plattform: Linux, python3 >= 3.4, Tensorflow >=1.2
 
 ## Python Packages: numpy, h5py, math, os, scipy.stats, cython, time, scikit-image, matplotlib, pickle
 
 To install those python packages run pip numpy, h5py, math, os, scipy.stats, cython, time, scikit-image, matplotlib, pickle.
 
 # Dataset

NUSTMS dataset has 5098 pairs of infrared and depth maps gathered by far-infrared camera and ranging radar on a driving vehicle. 
The equipment used to collect the data and an example of raw data are shown in the next.

![image](https://github.com/ivyharding999/Discrete-Convolutional-CRF-Networks-for-Depth-Estimation-from-Monocular-Infrared-Images/blob/master/Infrared%20images/data.png)

NUSTMS dataset  | infrared images and depth maps  | resolution(infrared images and depth maps)

 ---- | ----- | ------  

 training set  | 3488pair | 576×160,144×40

 validation set  | 586pair |  576×160,144×40
 
 testing set  | 1024pair |  576×160,144×40
  
 ## Dataset Link
  
Here we give the data link of its training set and test set, please click the link to get the infrared images and the corresponding ground-truth depths.
  
  [Baidu cloud disk](https://pan.baidu.com/s/1P8570lNk1JMvTTCARrDvaQ)
  
  [Google drive](https://drive.google.com/open?id=1z0AVvzpzGIiwWBpNqW-x4uh9OenDp5nn)
  
  [our own website](http://173.82.206.254/doku.php?id=public&do=#dokuwiki__top)

## ***Data reading method：***
```
def read_hdf5(file_name):
    with h5py.File(file_name, 'r') as f:
        images2 = np.asarray(f['images2'])
        images4 = np.asarray(f['images4'])
        depths = np.asarray(f['depths'])
    return images1,images2,images3,images4,depths
images1,images2,images3,images4,depths = read_hdf5('test_data.h5')
```
# Execute
## training step
## test step

# Citation 
If you benefit from this project, please consider citing our paper.

# TODO
•	 Build a Tensorflow 1.4 implementation
•	 Provide python 3 implementation

