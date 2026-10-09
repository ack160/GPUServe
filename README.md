# GPUServe

GPUServe is a C++/CUDA neural-network inference server that dynamically batches concurrent TCP requests and executes inference using custom GPU kernels.

The project starts from an MNIST classifier trained in PyTorch, exports the learned weights, implements inference manually in C++, and accelerates the same network using warp-parallel CUDA kernels.

## Architecture

GPUServe uses a multithreaded C++ TCP server with dynamic batching and custom CUDA inference.

    TCP Clients
         |
         v
    Thread-Safe Request Queue
         |
         v
    Dynamic Batch Assembly (max 8)
         |
         +----------------------+
         |                      |
         v                      v
    GPU Worker 0           GPU Worker 1
    Buffer Slot 0          Buffer Slot 1
    CUDA Stream 0          CUDA Stream 1
         |                      |
         +-----------+----------+
                     |
              Shared Weights
                     |
              CUDA Kernels
                     |
              Predictions

Each worker owns separate GPU buffers, pinned host memory, and a CUDA stream.

A mutex prevents simultaneous batch assembly while allowing separate workers to process batches independently.

GPU execution overlap has not been directly verified.

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

Incoming TCP requests enter a thread-safe queue. Batches contain up to 8 requests, with a maximum collection wait of 10 ms.

Profiling showed approximately 99.9-100% full batches under sustained 32-client load.

Two worker threads use independent buffer slots and CUDA streams. A batch-assembly mutex prevents competing workers from unnecessarily splitting batches.

## Asynchronous CUDA Pipeline

Each worker:

1. Packs images into pinned host memory.
2. Transfers inputs using cudaMemcpyAsync.
3. Runs custom warp-parallel CUDA kernels.
4. Transfers predictions using cudaMemcpyAsync.
5. Synchronizes its stream before reading results.
6. Sends each prediction to its corresponding client.

Separate CUDA streams enable independent work submission, but actual GPU execution overlap was not directly established.

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

## Historical CPU vs GPU Benchmark Results

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

## Dual-Worker Validation

Tests were performed on an NVIDIA A40.

### Correctness

- MNIST accuracy: 291/300 images (97.0%).
- Concurrent consistency: 900/900 predictions matched sequential reference predictions.
- Zero mismatches across three concurrent test rounds.
- Large repeated-image load tests completed without request failures.

The 97.0% accuracy applies only to the 300-image sample.

### Performance Comparison

| Configuration | Run 1 | Run 2 | Mean |
|---|---:|---:|---:|
| One worker | 43,287 req/s | 38,736 req/s | 41,012 req/s |
| Two workers | 49,733 req/s | 38,646 req/s | 44,190 req/s |

The dual-worker implementation showed approximately 7.7% higher mean throughput in this limited comparison.

Because results varied substantially, this is not evidence of a statistically reliable speedup.

Peak observed short-run throughput was approximately 59,000 requests/sec. This should not be interpreted as sustained throughput.

Simultaneous GPU execution across CUDA streams was not directly verified.

### Verification

Compile the response-checking tester:

    g++ -O3 -pthread load_test_verify.cpp -o load_test_verify

Check the sample digit under concurrent load:

    ./load_test_verify 32 10000

Check multiple MNIST images against sequential predictions:

    python3 verify_multi.py

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
