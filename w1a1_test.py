import bnn

#get
#!wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
#!wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
#unzip
#!gzip -d t10k-images-idx3-ubyte.gz
#!gzip -d t10k-labels-idx1-ubyte.gz

#read labels
print("Reading labels")
labels = []
with open("/home/xilinx/jupyter_notebooks/bnn/t10k-labels-idx1-ubyte","rb") as lbl_file:
    #read magic number and number of labels (MSB first) -> MNIST header
    magicNum = int.from_bytes(lbl_file.read(4), byteorder="big")
    countLbl = int.from_bytes(lbl_file.read(4), byteorder="big")
    #now the labels are following byte-wise
    for idx in range(countLbl):
        labels.append(int.from_bytes(lbl_file.read(1), byteorder="big"))
    lbl_file.close()
print("Initiating classifier")
lfcW1A1_classifier = bnn.LfcClassifier(bnn.NETWORK_LFCW1A1,"mnist",bnn.RUNTIME_HW)

print("Testing throughput")
result_W1A1 = lfcW1A1_classifier.classify_mnists("/home/xilinx/jupyter_notebooks/bnn/t10k-images-idx3-ubyte")