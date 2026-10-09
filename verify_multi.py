import urllib.request
import gzip
import struct
import array
import socket
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

N = 300
ROUNDS = 3
WORKERS = 32

base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
files = {
    "mnist.gz": "t10k-images-idx3-ubyte.gz",
    "mnist_labels.gz": "t10k-labels-idx1-ubyte.gz"
}

for local, remote in files.items():
    if not Path(local).exists():
        print("Downloading", remote)
        urllib.request.urlretrieve(base + remote, local)

with gzip.open("mnist.gz", "rb") as f:
    magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
    assert magic == 2051 and rows * cols == 784
    raw = f.read(N * 784)

with gzip.open("mnist_labels.gz", "rb") as f:
    magic, label_count = struct.unpack(">II", f.read(8))
    assert magic == 2049 and label_count == count
    labels = list(f.read(N))

assert len(raw) == N * 784 and len(labels) == N

images = []
for i in range(N):
    pixels = raw[i * 784:(i + 1) * 784]
    floats = array.array("f", (x / 255.0 for x in pixels))
    images.append(floats.tobytes())

def predict(image):
    with socket.create_connection(("127.0.0.1", 8080), timeout=10) as sock:
        sock.sendall(image)

        response = b""
        while len(response) < 4:
            chunk = sock.recv(4 - len(response))
            if not chunk:
                raise RuntimeError("Incomplete server response")
            response += chunk

        return struct.unpack("=i", response)[0]

print("Testing 300 images sequentially...")
reference = [predict(image) for image in images]

correct = sum(a == b for a, b in zip(reference, labels))

print("\n===== Sequential Reference =====")
print("Images:", N)
print("Ground-truth accuracy:", f"{correct}/{N} ({100*correct/N:.1f}%)")
print("Digits represented:", dict(sorted(Counter(labels).items())))
print("Predictions represented:", dict(sorted(Counter(reference).items())))

print("\n===== Concurrent Testing =====")

matches = 0
mismatches = []
concurrent_correct = 0

with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    for r in range(ROUNDS):
        results = list(pool.map(predict, images))
        matched = sum(a == b for a, b in zip(results, reference))
        matches += matched
        concurrent_correct += sum(a == b for a, b in zip(results, labels))

        for i, (actual, expected) in enumerate(zip(results, reference)):
            if actual != expected:
                mismatches.append((r + 1, i, expected, actual))

        print(f"Round {r+1}: {matched}/{N} matched reference")

print("\n===== FINAL RESULTS =====")
print("Total concurrent predictions:", N * ROUNDS)
print("Matching reference:", matches)
print("Mismatches:", len(mismatches))
print("Ground-truth accuracy under load:",
      f"{100*concurrent_correct/(N*ROUNDS):.1f}%")

if mismatches:
    print("First mismatches:", mismatches[:10])
else:
    print("PASS: All concurrent predictions matched sequential results.")
