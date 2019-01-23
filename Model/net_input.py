#!/usr/bin/env python3

import numpy as np
from numpy.random import randint
import h5py

def read_hdf5(file_name):
    with h5py.File(file_name, 'r') as f:
        images1 = np.asarray(f['images1'])
        images2 = np.asarray(f['images2'])
        images3 = np.asarray(f['images3'])
        images4 = np.asarray(f['images4'])
        depths = np.asarray(f['depths'])
    return images1,images2,images3,images4, depths

class DataSet(object):
    """data set class
    """
    def __init__(self, images1, images2, images3, images4, depths):
        self._images1 = images1
        self._images2 = images2
        self._images3 = images3
        self._images4 = images4
        self._depths = depths
        self._batch_index = 0

    @property
    def images1(self):
        return self._images1

    @property
    def images2(self):
        return self._images2

    @property
    def images3(self):
        return self._images3

    @property
    def images4(self):
        return self._images4

    @property
    def depths(self):
        return self._depths
    
    def _shuffle(self):
        index = np.arange(self.images1.shape[0])
        np.random.shuffle(index)
        self._images1 = self._images1[index]
        self._images2 = self._images2[index]
        self._images3 = self._images3[index]
        self._images4 = self._images4[index]
        self._depths = self._depths[index]

    def next_batch(self, batch_size):
        if self._batch_index >= int(self._images1.shape[0] / batch_size):
            self._shuffle()
            self._batch_index = 0
        ran = np.arange(self._batch_index * batch_size,
                            (self._batch_index + 1) * batch_size)
        self._batch_index += 1
        return self._images1[ran], self._images2[ran], self._images3[ran], self._images4[ran], self._depths[ran]

    def batch(self, batch_size):
        index = randint(0, self._images1.shape[0], batch_size)
        return self._images1[index], self._images2[index],self._images3[index],self._images4[index], self._depths[index]



def read_file(file_name):
    print('reading file ...')
    with h5py.File(file_name, 'r') as f:
        images1 = np.asarray(f['images1'])
        images2 = np.asarray(f['images2'])
        images3 = np.asarray(f['images3'])
        images4 = np.asarray(f['images4'])
        depths = np.asarray(f['depths'])
        im1_shape = images1.shape
        im2_shape = images2.shape
        im3_shape = images3.shape
        im4_shape = images4.shape
        de_shape = depths.shape
    images1 = np.reshape(images1, (im1_shape[0], im1_shape[1], im1_shape[2], 1))
    images2 = np.reshape(images2, (im2_shape[0], im2_shape[1], im2_shape[2], 1))
    images3 = np.reshape(images3, (im3_shape[0], im3_shape[1], im3_shape[2], 1))
    images4 = np.reshape(images4, (im4_shape[0], im4_shape[1], im4_shape[2], 1))
    depths = np.reshape(depths.astype('int32'), (de_shape[0], de_shape[1], de_shape[2], 1))
    dataset = DataSet(images1, images2, images3, images4, depths)
    dataset._shuffle()
    print('Reading done.')
    return dataset
