# simpel code showing how to decode mnist raw format
# download mnist:
# wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
# wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
# wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
# wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
# put downloaded files to ./bk0/data folder
import gzip
import numpy as np
import matplotlib.pyplot as plt

images_file = gzip.open('./bk0/data/train-images-idx3-ubyte.gz','r')
labels_file = gzip.open('./bk0/data/train-labels-idx1-ubyte.gz','r')

image_size = 28
num_images = 60_000

# DECODE images file
# skip images file header
# image file header description
# first 4 bytes is a magic number
# second 4 bytes is the number of images
# third 4 bytes is the row count
# fourth 4 bytes is the column count
# rest is the image pixel data, each pixel is stored as an unsigned byte
# pixel values are 0 to 255
images_file.read(16)
# read images bytes
imgBuf = images_file.read(image_size * image_size * num_images)
images = np.frombuffer(imgBuf, dtype=np.uint8).astype(np.float32)
images = images.reshape(num_images, image_size, image_size, 1)

# DECODE labels file
# skip labels file header
# first 4 bytes is a magic number
# second 4 bytes is the number of labels
# rest is the label data, each label is stored as unsigned byte
# label values are 0 to 9
labels_file.read(8)
# read labels bytes
lbBuf = labels_file.read(num_images)
labels = np.frombuffer(lbBuf, dtype=np.uint8).astype(np.int64)

# show some result
imgNum = 1
print("image number: ", imgNum)
print("image label: ", labels[imgNum])
# show image
image = np.asarray(images[imgNum]).squeeze()
plt.imshow(image)
plt.show()
