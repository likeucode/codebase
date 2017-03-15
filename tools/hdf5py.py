#read
import h5py
f = h5py.File('mytestfile.hdf5', 'r')
f.keys()
dset = f['mydataset']
dset.shape


#write
import h5py
import numpy as np
f = h5py.File("mytestfile.hdf5", "w")
dset = f.create_dataset("mydataset", (100,), dtype='i')

arr = np.arange(100)
dset = f.create_dataset("init", data=arr)
dset = f.create_dataset("chunked", (1000, 1000), chunks=(100, 100))

