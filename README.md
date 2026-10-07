# GPUServe

GPUServe is a C++/CUDA neural-network inference server that dynamically batches concurrent TCP requests and executes inference using custom GPU kernels.

The project starts from an MNIST classifier trained in PyTorch, exports the learned weights, implements inference manually in C++, and accelerates the same network using warp-parallel CUDA kernels.

## Architecture

```text
Clients
   |
   | TCP requests
   v
+---------------------+
|   C++ TCP Server    |
+---------------------+
          |
          v
+---------------------+
| Thread-Safe Queue   |
+---------------------+
          |
          v
+---------------------+
|  Dynamic Batcher    |
| max batch = 8       |
| wait <= 10 ms       |
+---------------------+
          |
          +-------------------+
          |                   |
          v                   v
   CPU Backend          CUDA Backend
   C++ inference        Warp-parallel kernels
          |                   |
          +---------+---------+
                    |
                    v
              Predictions
                    |
                    v
                 Clients
```

## Model

The model is a fully connected MNIST classifier:

```text
28x28 image
    |
    v
784 inputs
    |
    v
Linear(784 -> 128)
    |
    v
ReLU
    |
    v
Linear(128 -> 10)
    |
    v
digit prediction
```

The trained model weights are exported into binary files under `model_data/`.

## CUDA Implementation

The GPU inference backend is implemented directly in CUDA rather than using PyTorch at serving time.

For the first layer, each warp cooperatively computes a hidden-neuron dot product across the 784 input pixels. Threads accumulate partial sums and combine them using warp shuffle reductions.

The second layer uses the same approach across the 128 hidden activations to produce the 10 output scores.

## Dynamic Batching

Incoming TCP requests are placed into a thread-safe queue.

The batcher:

- takes the first available request
- waits up to 10 ms for nearby requests
- collects up to 8 requests
- performs inference on the batch
- returns each prediction to the correct client

At low traffic, the batch wait can slightly increase latency. At higher concurrency, batches fill quickly and the GPU can process requests much more efficiently.

## Benchmark Methodology

CPU and GPU backends were tested on the same Runpod machine using:

- the same neural network
- the same TCP server design
- the same dynamic batching policy
- the same load-testing client
- an NVIDIA A40 for the CUDA backend

The load tester measures full end-to-end request latency:

```text
TCP connect
-> send image
-> request queue
-> dynamic batching
-> inference
-> receive prediction
-> close connection
```

## Benchmark Results

| Concurrent Clients | CPU Throughput | GPU Throughput | Speedup | CPU Avg Latency | GPU Avg Latency |
|---:|---:|---:|---:|---:|---:|
| 1 | 90.7 req/s | 93.5 req/s | 1.03x | 11.02 ms | 10.69 ms |
| 8 | 2,695 req/s | 12,535 req/s | 4.65x | 2.96 ms | 0.63 ms |
| 16 | 6,398 req/s | 13,501 req/s | 2.11x | 2.49 ms | 0.74 ms |
| 32 | 6,509 req/s | 30,511 req/s | 4.69x | 4.89 ms | 0.82 ms |

At 32 concurrent clients, the CUDA backend achieved:

- **30.5K requests/sec**
- **4.69x higher throughput**
- **0.82 ms average end-to-end latency**
- **83% lower average latency** than the CPU backend

Tail latency at 32 clients:

```text
CPU p95: 5.07 ms
GPU p95: 1.17 ms
```

That is approximately a 77% reduction in p95 latency.

## Build and Run

### CPU Server

```bash
g++ -O3 -pthread server.cpp -o server
./server
```

### CUDA Server

For an NVIDIA Ampere GPU such as the A40:

```bash
nvcc -O3 -arch=sm_86 -Xcompiler -pthread server_gpu.cu -o server_gpu
./server_gpu
```

### Client

```bash
g++ -O3 client.cpp -o client
./client
```

### Load Tester

```bash
g++ -O3 -pthread load_test.cpp -o load_test
./load_test 32 100
```

`./load_test 32 100` creates 32 concurrent client threads with 100 requests per thread, for 3,200 total requests.

## Project Structure

```text
GPUServe/
├── server.cpp
├── server_gpu.cu
├── client.cpp
├── load_test.cpp
├── cpu_inference.cpp
├── gpu_inference.cu
├── batch_inference.cu
├── batch_inference_warp.cu
├── benchmark.cu
├── benchmark_warp.cu
├── get_sample.py
└── model_data/
    ├── W1.bin
    ├── b1.bin
    ├── W2.bin
    └── b2.bin
```

## Technologies

C++, CUDA, PyTorch, Linux, TCP sockets, multithreading, mutexes, condition variables, dynamic batching, warp-level GPU primitives, and performance benchmarking.

## Key Result

GPUServe sustained approximately **30.5K inference requests/sec** at 32 concurrent clients using custom CUDA kernels, delivering approximately **4.7x the throughput** and **83% lower average latency** than the CPU backend on the same machine.
