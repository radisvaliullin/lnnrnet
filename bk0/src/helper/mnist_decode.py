# mnist_decode.py
import gzip
import numpy as np

# images 60k (training data)
IMAGES_60K_FILE_PATH = './bk0/data/train-images-idx3-ubyte.gz'
LABELS_60K_FILE_PATH = './bk0/data/train-labels-idx1-ubyte.gz'
# images 10k (test data)
IMAGES_10K_FILE_PATH = './bk0/data/t10k-images-idx3-ubyte.gz'
LABELS_10K_FILE_PATH = './bk0/data/t10k-labels-idx1-ubyte.gz'

# image size 28x28
IMAGE_SIZE = 28
# large set 60k
NUM_IMAGES_60K = 60_000
NUM_IMAGES_50K = 50_000
NUM_IMAGES_10K = 10_000

def mnist_decode():
    """
    open and decode mnist gzip files

    split 60k images to 50k training data and 10k validation data
    and return 50k as train data and 10k as valid data
    return 10k images as test data
    """
    # open gzip files
    images_60k_file = gzip.open(IMAGES_60K_FILE_PATH,'r')
    labels_60k_file = gzip.open(LABELS_60K_FILE_PATH,'r')
    images_10k_file = gzip.open(IMAGES_10K_FILE_PATH,'r')
    labels_10k_file = gzip.open(LABELS_10K_FILE_PATH,'r')

    # decode content to numpy arrays
    # skip headers
    images_60k_file.read(16)
    labels_60k_file.read(8)
    images_10k_file.read(16)
    labels_10k_file.read(8)
    # read data bytes
    # 60k images
    img_60k_buf = images_60k_file.read(IMAGE_SIZE * IMAGE_SIZE * NUM_IMAGES_60K)
    img_60k_arr = np.frombuffer(img_60k_buf, dtype=np.uint8).astype(np.float32)
    img_60k = img_60k_arr.reshape(NUM_IMAGES_60K, IMAGE_SIZE*IMAGE_SIZE)
    img_60k = img_60k/256.0
    # 60k labels
    lb_60k_buf = labels_60k_file.read(NUM_IMAGES_60K)
    lb_60k = np.frombuffer(lb_60k_buf, dtype=np.uint8).astype(np.int64)
    # 10k images
    img_10k_buf = images_10k_file.read(IMAGE_SIZE * IMAGE_SIZE * NUM_IMAGES_10K)
    img_10k_arr = np.frombuffer(img_10k_buf, dtype=np.uint8).astype(np.float32)
    img_10k = img_10k_arr.reshape(NUM_IMAGES_10K, IMAGE_SIZE*IMAGE_SIZE)
    img_10k = img_10k/256.0
    # 10k labels
    lb_10k_buf = labels_10k_file.read(NUM_IMAGES_10K)
    lb_10k = np.frombuffer(lb_10k_buf, dtype=np.uint8).astype(np.int64)

    # compose train_data, valid_data, test_data
    # 50k as train data
    train_data = (img_60k[0:NUM_IMAGES_50K], lb_60k[0:NUM_IMAGES_50K])
    # 10k as valid data
    valid_data = (img_60k[NUM_IMAGES_50K:], lb_60k[NUM_IMAGES_50K:])
    # test data
    test_data = (img_10k, lb_10k)

    return train_data, valid_data, test_data

def mnist_decode_reformat():
    """
    Reformat mnist decoded data. 

    Reformated structs:
    train_data - list of 50,000 tuples (x, y), where
    x is 784-dimensional numpy.ndarray input image,
    y is 10-dimensional numpy.ndarray unit vector corresponding to the correct digit for x

    valid_data and test_data - lists 10,000 tuples (x, y), where
    x is 784-dimensional numpy.ndarry input image,
    y is digit values (integers) for x.
    """
    tr_d, va_d, te_d = mnist_decode()
    train_in = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    train_res = [vectoriz_result(y) for y in tr_d[1]]
    train_data = list(zip(train_in, train_res))
    valid_in = [np.reshape(x, (784, 1)) for x in va_d[0]]
    valid_data = list(zip(valid_in, va_d[1]))
    test_in = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_in, te_d[1]))
    return (train_data, valid_data, test_data)

def vectoriz_result(j):
    """
    Convert digit (0-9) to 10-dimensional vector.
    Where jth position has 1.0 value (over zero) representing corresponding digit.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
