
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

using namespace std;

template <typename T>
vector<T> loadFile(const string& filename, int count) {
    vector<T> data(count);

    ifstream file(filename, ios::binary);
    file.read(reinterpret_cast<char*>(data.data()),
              count * sizeof(T));

    return data;
}


// -------------------------------------------------
// Layer 1
// One CUDA block = one image
// One CUDA thread = one hidden neuron
// -------------------------------------------------

__global__ void layer1Batch(
    const float* W1,
    const float* b1,
    const float* images,
    float* hidden,
    int batchSize
) {
    int imageIndex = blockIdx.x;
    int h = threadIdx.x;

    if (imageIndex < batchSize && h < 128) {

        const float* x =
            images + imageIndex * 784;

        float sum = b1[h];

        for (int i = 0; i < 784; i++) {
            sum += W1[h * 784 + i] * x[i];
        }

        hidden[imageIndex * 128 + h] =
            sum > 0.0f ? sum : 0.0f;
    }
}


// -------------------------------------------------
// Layer 2
// One CUDA block = one image
// One CUDA thread = one digit score
// -------------------------------------------------

__global__ void layer2Batch(
    const float* W2,
    const float* b2,
    const float* hidden,
    float* scores,
    int batchSize
) {
    int imageIndex = blockIdx.x;
    int o = threadIdx.x;

    if (imageIndex < batchSize && o < 10) {

        float sum = b2[o];

        for (int h = 0; h < 128; h++) {
            sum +=
                W2[o * 128 + h] *
                hidden[imageIndex * 128 + h];
        }

        scores[imageIndex * 10 + o] = sum;
    }
}


int main() {

    const int BATCH_SIZE = 128;
    const int INPUT_SIZE = 784;
    const int HIDDEN_SIZE = 128;
    const int OUTPUT_SIZE = 10;


    // --------------------------
    // Load model + test batch
    // --------------------------

    vector<float> W1 =
        loadFile<float>(
            "model_data/W1.bin",
            HIDDEN_SIZE * INPUT_SIZE
        );

    vector<float> b1 =
        loadFile<float>(
            "model_data/b1.bin",
            HIDDEN_SIZE
        );

    vector<float> W2 =
        loadFile<float>(
            "model_data/W2.bin",
            OUTPUT_SIZE * HIDDEN_SIZE
        );

    vector<float> b2 =
        loadFile<float>(
            "model_data/b2.bin",
            OUTPUT_SIZE
        );

    vector<float> images =
        loadFile<float>(
            "model_data/batch_images.bin",
            BATCH_SIZE * INPUT_SIZE
        );

    vector<int> labels =
        loadFile<int>(
            "model_data/batch_labels.bin",
            BATCH_SIZE
        );


    // --------------------------
    // Allocate GPU memory
    // --------------------------

    float *d_W1, *d_b1;
    float *d_W2, *d_b2;
    float *d_images;
    float *d_hidden;
    float *d_scores;

    cudaMalloc(&d_W1,
        W1.size() * sizeof(float));

    cudaMalloc(&d_b1,
        b1.size() * sizeof(float));

    cudaMalloc(&d_W2,
        W2.size() * sizeof(float));

    cudaMalloc(&d_b2,
        b2.size() * sizeof(float));

    cudaMalloc(&d_images,
        images.size() * sizeof(float));

    cudaMalloc(&d_hidden,
        BATCH_SIZE * HIDDEN_SIZE * sizeof(float));

    cudaMalloc(&d_scores,
        BATCH_SIZE * OUTPUT_SIZE * sizeof(float));


    // --------------------------
    // Copy data to GPU
    // --------------------------

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

    cudaMemcpy(
        d_images,
        images.data(),
        images.size() * sizeof(float),
        cudaMemcpyHostToDevice
    );


    // --------------------------
    // Run entire batch
    // --------------------------

    layer1Batch<<<BATCH_SIZE, 128>>>(
        d_W1,
        d_b1,
        d_images,
        d_hidden,
        BATCH_SIZE
    );

    layer2Batch<<<BATCH_SIZE, 32>>>(
        d_W2,
        d_b2,
        d_hidden,
        d_scores,
        BATCH_SIZE
    );


    // --------------------------
    // Copy results back
    // --------------------------

    vector<float> scores(
        BATCH_SIZE * OUTPUT_SIZE
    );

    cudaMemcpy(
        scores.data(),
        d_scores,
        scores.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    );


    // --------------------------
    // Calculate predictions
    // --------------------------

    int correct = 0;

    for (int image = 0;
         image < BATCH_SIZE;
         image++) {

        int prediction = 0;

        for (int digit = 1;
             digit < OUTPUT_SIZE;
             digit++) {

            if (
                scores[image * 10 + digit] >
                scores[image * 10 + prediction]
            ) {
                prediction = digit;
            }
        }

        if (prediction == labels[image]) {
            correct++;
        }

        if (image < 20) {
            cout
                << "Image " << image
                << " | actual: " << labels[image]
                << " | predicted: " << prediction
                << endl;
        }
    }


    float accuracy =
        100.0f * correct / BATCH_SIZE;

    cout << endl;
    cout << "Batch accuracy: "
         << accuracy << "%"
         << endl;


    // --------------------------
    // Cleanup
    // --------------------------

    cudaFree(d_W1);
    cudaFree(d_b1);
    cudaFree(d_W2);
    cudaFree(d_b2);
    cudaFree(d_images);
    cudaFree(d_hidden);
    cudaFree(d_scores);

    return 0;
}
