# Discrete Convolutional CRF Networks for Depth Estimation from Monocular Infrared Images
   This paper focuses on depth estimation from monocular infrared images, which is essential for understanding the structure of 3D scene and can promote the development of night vision applications. 
  
  
  NUSTMS dataset has 5098 pairs of infrared and depth maps gathered by far-infrared camera and ranging radar on a driving vehicle. The resolution of infrared image and corresponding depth map are 576×160 and 144×40 respectively. In our experiments, the dataset is split into a training set (3488 pairs), a validation set (586 pairs) and a test set (1024 pairs).
  
  The infrared images,corresponding ground-truth depths and depth estimation results are shown below:
  
  ![image](https://github.com/ivyharding999/Discrete-Convolutional-CRF-Networks-for-Depth-Estimation-from-Monocular-Infrared-Images/blob/master/Infrared%20images/data.png)
  
  
  Here we give the data link of its training set and test set, please click the link to get the infrared images and the corresponding ground-truth depths.
  
  [Baidu cloud disk](https://pan.baidu.com/s/1P8570lNk1JMvTTCARrDvaQ)
  
  [our own website](http://173.82.206.254/doku.php?id=public&do=#dokuwiki__top)

***Data reading method：***
```
def read_hdf5(file_name):
    with h5py.File(file_name, 'r') as f:
        images2 = np.asarray(f['images2'])
        images4 = np.asarray(f['images4'])
        depths = np.asarray(f['depths'])
    return images1,images2,images3,images4,depths
images1,images2,images3,images4,depths = read_hdf5('test_data.h5')
```
