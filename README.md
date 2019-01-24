
# Discrete Convolutional CRF Networks for Depth Estimation from Monocular Infrared Images

This repository contains the reference implementation for our proposed DCCRF networks in Tensorflow.

This paper focuses on depth estimation from monocular infrared images, which is essential for understanding the structure of 3D scene and can promote the development of night vision applications. 

Examples of predicted depth maps on NUSTMS dataset: Infrared Image (top row), Ground-Truth Depth (center) and Predicted Depth Maps (bottom row).

![image](https://github.com/ivyharding999/Discrete-Convolutional-CRF-Networks-for-Depth-Estimation-from-Monocular-Infrared-Images/blob/master/Infrared%20images/DATA.png)


### Click the link to get some examples of predicted depth maps on NUSTMS dataset:


[Results](https://pan.baidu.com/s/1P8570lNk1JMvTTCARrDvaQ)

 # Requirements
 
 ### Plattform 
 ```
 Linux, python3 >= 3.4, Tensorflow >=1.2
 ```
 ### Python Packages 
 ```
 numpy, h5py, math, os, scipy.stats, cython, time, scikit-image, matplotlib, pickle
 ```
 To install those python packages run
 ```
 pip install numpy, h5py, math, os, scipy.stats, cython, time, scikit-image, matplotlib, pickle.
 ```
 # Model

The figure shows the proposed DCCRF, which has two components, one of them for feature learning and another for CRF loss layer. 

 ![image](https://github.com/ivyharding999/Discrete-Convolutional-CRF-Networks-for-Depth-Estimation-from-Monocular-Infrared-Images/blob/master/Infrared%20images/Fig1.png)
 
 
 # Dataset

NUSTMS dataset has 5098 pairs of infrared and depth maps gathered by far-infrared camera and ranging radar on a driving vehicle. 
The equipment used to collect the data and an example of raw data are shown below.

![image](https://github.com/ivyharding999/Discrete-Convolutional-CRF-Networks-for-Depth-Estimation-from-Monocular-Infrared-Images/blob/master/Infrared%20images/Fig6.png)



### Data structure
```

   NUSTMS         | Infrared images   | Ground-Truth Depth
 --------------   | ----------------- | ---------------
 Training set     |   3488x576×160    |  3488x144×40
 Validation set   |   586x576×160     |  586x144×40 
 Testing set      |   1024x576×160    |  1024x144×40    

``` 

### Dataset Link
 
Here we give the data link of its training set and test set, please click the link to get the infrared images and the corresponding ground-truth depths.


  
  [NUSTMS data / Baidu SkyDrive](https://pan.baidu.com/s/1P8570lNk1JMvTTCARrDvaQ)
  
  [NUSTMS data / Google drive](https://drive.google.com/open?id=1z0AVvzpzGIiwWBpNqW-x4uh9OenDp5nn)


### Data reading method：
```
def read_hdf5(file_name):
    with h5py.File(file_name, 'r') as f:
        images1 = np.asarray(f['images2'])
        images2 = np.asarray(f['images4'])
        depths = np.asarray(f['depths'])
    return images1,images2,depths
images1,images2,depths = read_hdf5('test_data.h5')
```

# Execute

### training stage
```
python run train_scale3.py
```


### test stage
```
python run test_scale3.py
```

# Citation 
```
If you benefit from this project, please consider citing our paper.
```

# TODO

•	 Build a Tensorflow 1.4 implementation

•	 Provide python 3 implementation

