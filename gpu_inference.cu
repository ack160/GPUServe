
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

using namespace std;

vector<float> loadFile(const string& filename, int count) {
    vector<float> data(count);

    ifstream file(filename, ios::binary);
    file.read(reinterpret_cast<char*>(data.data()),
              count * sizeof(float));

    return data;
}

// Layer 1: 784 inputs -> 128 hidden neurons
__global__ void layer1(
    const float* W1,
    const float* b1,
    const float* x,
    float* hidden
) {
    int h = threadIdx.x;

    if (h < 128) {

        float sum = b1[h];

        for (int i = 0; i < 784; i++) {
            sum += W1[h * 784 + i] * x[i];
        }

        // ReLU
        hidden[h] = sum > 0.0f ? sum : 0.0f;
    }
}


// Layer 2: 128 hidden neurons -> 10 digit scores
__global__ void layer2(
    const float* W2,
    const float* b2,
    const float* hidden,
    float* scores
) {
    int o = threadIdx.x;

    if (o < 10) {

        float sum = b2[o];

        for (int h = 0; h < 128; h++) {
            sum += W2[o * 128 + h] * hidden[h];
        }

        scores[o] = sum;
    }
}


int main() {

    const int INPUT_SIZE = 784;
    const int HIDDEN_SIZE = 128;
    const int OUTPUT_SIZE = 10;

    // Load files on CPU
    vector<float> W1 =
        loadFile("model_data/W1.bin", HIDDEN_SIZE * INPUT_SIZE);

    vector<float> b1 =
        loadFile("model_data/b1.bin", HIDDEN_SIZE);

    vector<float> W2 =
        loadFile("model_data/W2.bin", OUTPUT_SIZE * HIDDEN_SIZE);

    vector<float> b2 =
        loadFile("model_data/b2.bin", OUTPUT_SIZE);

    vector<float> x =
        loadFile("model_data/sample_image.bin", INPUT_SIZE);


    // GPU pointers
    float *d_W1, *d_b1;
    float *d_W2, *d_b2;
    float *d_x;
    float *d_hidden;
    float *d_scores;


    // Allocate GPU memory
    cudaMalloc(&d_W1, W1.size() * sizeof(float));
    cudaMalloc(&d_b1, b1.size() * sizeof(float));

    cudaMalloc(&d_W2, W2.size() * sizeof(float));
    cudaMalloc(&d_b2, b2.size() * sizeof(float));

    cudaMalloc(&d_x, x.size() * sizeof(float));

    cudaMalloc(&d_hidden, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_scores, OUTPUT_SIZE * sizeof(float));


    // Copy CPU data -> GPU
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
        d_x,
        x.data(),
        x.size() * sizeof(float),
        cudaMemcpyHostToDevice
    );


    // Run neural network on GPU
    layer1<<<1, 128>>>(
        d_W1,
        d_b1,
        d_x,
        d_hidden
    );

    layer2<<<1, 10>>>(
        d_W2,
        d_b2,
        d_hidden,
        d_scores
    );


    // Copy scores GPU -> CPU
    vector<float> scores(OUTPUT_SIZE);

    cudaMemcpy(
        scores.data(),
        d_scores,
        OUTPUT_SIZE * sizeof(float),
        cudaMemcpyDeviceToHost
    );


    // Find prediction
    int prediction = 0;

    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (scores[i] > scores[prediction]) {
            prediction = i;
        }
    }


    // Print results
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        cout << "Digit " << i
             << " score: "
             << scores[i]
             << endl;
    }

    cout << endl;
    cout << "CUDA prediction: "
         << prediction
         << endl;


    // Free GPU memory
    cudaFree(d_W1);
    cudaFree(d_b1);
    cudaFree(d_W2);
    cudaFree(d_b2);
    cudaFree(d_x);
    cudaFree(d_hidden);
    cudaFree(d_scores);

    return 0;
}
