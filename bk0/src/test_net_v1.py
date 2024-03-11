from helper.mnist_decode import mnist_decode_reformat
from network_v1 import Network

trd, vad, ted = mnist_decode_reformat()

net = Network([784, 30, 10])
net.SGD(trd, 30, 10, 3.0, test_data=ted)
