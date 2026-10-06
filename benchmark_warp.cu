
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>

using namespace std;

template <typename T>
vector<T> loadFile(const string& filename, int count) {
    vector<T> data(count);

    ifstream file(filename, ios::binary);

    if (!file) {
        cerr << "ERROR: Could not open " << filename << endl;
        exit(1);
    }

    file.read(
        reinterpret_cast<char*>(data.data()),
        count * sizeof(T)
    );

    if (!file) {
        cerr << "ERROR: Failed reading " << filename << endl;
        exit(1);
    }

    return data;
}


// =====================================================
// CUDA kernels
// =====================================================


__global__ void layer1Batch(
    const float* W1,
    const float* b1,
    const float* images,
    float* hidden,
    int batchSize
) {
    // 16 blocks per image
    // 8 warps per block
    // 1 warp computes 1 hidden neuron

    int image = blockIdx.x / 16;
    int blockWithinImage = blockIdx.x % 16;

    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    int h = blockWithinImage * 8 + warp;

    if (image >= batchSize) return;

    const float* x = images + image * 784;
    const float* weights = W1 + h * 784;

    float sum = 0.0f;

    // 32 lanes split the 784-element dot product
    for (int i = lane; i < 784; i += 32) {
        sum += weights[i] * x[i];
    }

    // Combine the 32 partial sums inside the warp
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Lane 0 now holds the complete dot product
    if (lane == 0) {
        sum += b1[h];

        hidden[image * 128 + h] =
            sum > 0.0f ? sum : 0.0f;
    }
}


__global__ void layer2Batch(
    const float* W2,
    const float* b2,
    const float* hidden,
    float* scores,
    int batchSize
) {
    // 1 block per image
    // 10 warps = 10 output digits

    int image = blockIdx.x;

    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    if (image >= batchSize || warp >= 10) return;

    int o = warp;

    float sum = 0.0f;

    for (int h = lane; h < 128; h += 32) {
        sum +=
            W2[o * 128 + h] *
            hidden[image * 128 + h];
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        scores[image * 10 + o] =
            sum + b2[o];
    }
}


// =====================================================
// CPU inference
// =====================================================

void cpuInference(
    const vector<float>& W1,
    const vector<float>& b1,
    const vector<float>& W2,
    const vector<float>& b2,
    const vector<float>& images,
    int batchSize,
    vector<float>& scores
) {

    vector<float> hidden(batchSize * 128);

    for (int image = 0; image < batchSize; image++) {

        const float* x =
            images.data() + image * 784;

        for (int h = 0; h < 128; h++) {

            float sum = b1[h];

            for (int i = 0; i < 784; i++) {
                sum +=
                    W1[h * 784 + i] *
                    x[i];
            }

            hidden[image * 128 + h] =
                sum > 0.0f ? sum : 0.0f;
        }


        for (int o = 0; o < 10; o++) {

            float sum = b2[o];

            for (int h = 0; h < 128; h++) {
                sum +=
                    W2[o * 128 + h] *
                    hidden[image * 128 + h];
            }

            scores[image * 10 + o] = sum;
        }
    }
}


// =====================================================
// Main benchmark
// =====================================================

int main() {

    const int MAX_BATCH = 128;
    const int REPEATS = 50;

    vector<int> batchSizes =
        {1, 8, 16, 32, 64, 128};


    // Load model
    vector<float> W1 =
        loadFile<float>(
            "model_data/W1.bin",
            128 * 784
        );

    vector<float> b1 =
        loadFile<float>(
            "model_data/b1.bin",
            128
        );

    vector<float> W2 =
        loadFile<float>(
            "model_data/W2.bin",
            10 * 128
        );

    vector<float> b2 =
        loadFile<float>(
            "model_data/b2.bin",
            10
        );


    // Load 128 test images
    vector<float> images =
        loadFile<float>(
            "model_data/batch_images.bin",
            MAX_BATCH * 784
        );


    // =================================================
    // GPU setup
    // =================================================

    float *d_W1, *d_b1;
    float *d_W2, *d_b2;
    float *d_images;
    float *d_hidden;
    float *d_scores;

    cudaMalloc(&d_W1, W1.size() * sizeof(float));
    cudaMalloc(&d_b1, b1.size() * sizeof(float));
    cudaMalloc(&d_W2, W2.size() * sizeof(float));
    cudaMalloc(&d_b2, b2.size() * sizeof(float));

    cudaMalloc(
        &d_images,
        MAX_BATCH * 784 * sizeof(float)
    );

    cudaMalloc(
        &d_hidden,
        MAX_BATCH * 128 * sizeof(float)
    );

    cudaMalloc(
        &d_scores,
        MAX_BATCH * 10 * sizeof(float)
    );


    // Model weights stay on GPU
    cudaMemcpy(
        d_W1,
        W1.data(),
        W1.size() * sizeof(float),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_b1,
        b1.data(),
        b1.size() * sizeof(float),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_W2,
        W2.data(),
        W2.size() * sizeof(float),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_b2,
        b2.data(),
        b2.size() * sizeof(float),
        cudaMemcpyHostToDevice
    );


    cout << fixed << setprecision(4);

    cout
        << "Batch\tCPU ms\tGPU Kernel ms\tGPU Total ms\tGPU img/sec"
        << endl;


    // =================================================
    // Test each batch size
    // =================================================

    for (int batchSize : batchSizes) {

        vector<float> cpuScores(batchSize * 10);
        vector<float> gpuScores(batchSize * 10);


        // ---------------------------------------------
        // CPU timing
        // ---------------------------------------------

        auto cpuStart =
            chrono::high_resolution_clock::now();

        for (int r = 0; r < REPEATS; r++) {
            cpuInference(
                W1,
                b1,
                W2,
                b2,
                images,
                batchSize,
                cpuScores
            );
        }

        auto cpuEnd =
            chrono::high_resolution_clock::now();

        double cpuMs =
            chrono::duration<double, milli>(
                cpuEnd - cpuStart
            ).count() / REPEATS;


        // ---------------------------------------------
        // GPU warmup
        // ---------------------------------------------

        cudaMemcpy(
            d_images,
            images.data(),
            batchSize * 784 * sizeof(float),
            cudaMemcpyHostToDevice
        );

        layer1Batch<<<batchSize * 16, 256>>>(
            d_W1,
            d_b1,
            d_images,
            d_hidden,
            batchSize
        );

        layer2Batch<<<batchSize, 320>>>(
            d_W2,
            d_b2,
            d_hidden,
            d_scores,
            batchSize
        );

        cudaDeviceSynchronize();


        // ---------------------------------------------
        // GPU kernel timing
        // ---------------------------------------------

        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float totalKernelMs = 0.0f;

        for (int r = 0; r < REPEATS; r++) {

            cudaEventRecord(start);

            layer1Batch<<<batchSize * 16, 256>>>(
                d_W1,
                d_b1,
                d_images,
                d_hidden,
                batchSize
            );

            layer2Batch<<<batchSize, 320>>>(
                d_W2,
                d_b2,
                d_hidden,
                d_scores,
                batchSize
            );

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms;
            cudaEventElapsedTime(
                &ms,
                start,
                stop
            );

            totalKernelMs += ms;
        }

        double kernelMs =
            totalKernelMs / REPEATS;


        // ---------------------------------------------
        // GPU total timing
        // H2D + kernels + D2H
        // ---------------------------------------------

        auto gpuStart =
            chrono::high_resolution_clock::now();

        for (int r = 0; r < REPEATS; r++) {

            cudaMemcpy(
                d_images,
                images.data(),
                batchSize * 784 * sizeof(float),
                cudaMemcpyHostToDevice
            );

            layer1Batch<<<batchSize * 16, 256>>>(
                d_W1,
                d_b1,
                d_images,
                d_hidden,
                batchSize
            );

            layer2Batch<<<batchSize, 320>>>(
                d_W2,
                d_b2,
                d_hidden,
                d_scores,
                batchSize
            );

            cudaMemcpy(
                gpuScores.data(),
                d_scores,
                batchSize * 10 * sizeof(float),
                cudaMemcpyDeviceToHost
            );
        }

        auto gpuEnd =
            chrono::high_resolution_clock::now();

        double gpuTotalMs =
            chrono::duration<double, milli>(
                gpuEnd - gpuStart
            ).count() / REPEATS;


        double gpuThroughput =
            batchSize / (gpuTotalMs / 1000.0);


        cout
            << batchSize << "\t"
            << cpuMs << "\t"
            << kernelMs << "\t\t"
            << gpuTotalMs << "\t\t"
            << gpuThroughput
            << endl;


        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }


    // =================================================
    // Cleanup
    // =================================================

    cudaFree(d_W1);
    cudaFree(d_b1);
    cudaFree(d_W2);
    cudaFree(d_b2);
    cudaFree(d_images);
    cudaFree(d_hidden);
    cudaFree(d_scores);

    return 0;
}
