import urllib.request
import gzip
import struct
import array

url = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz"

urllib.request.urlretrieve(url, "mnist.gz")

with gzip.open("mnist.gz", "rb") as f:
    magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
    pixels = f.read(rows * cols)

values = [p / 255.0 for p in pixels]

with open("sample_image.bin", "wb") as out:
    array.array("f", values).tofile(out)

print("Saved first MNIST test image — actual digit is 7")
