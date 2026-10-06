
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

using namespace std;

vector<float> loadFile(const string& filename, int count) {
    vector<float> data(count);

    ifstream file(filename, ios::binary);
    file.read(reinterpret_cast<char*>(data.data()),
              count * sizeof(float));

    return data;
}

int main() {
    const int INPUT_SIZE = 784;
    const int HIDDEN_SIZE = 128;
    const int OUTPUT_SIZE = 10;

    // Load weights, biases, and test image
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


    // -------------------------
    // Layer 1: 784 -> 128
    // -------------------------

    vector<float> hidden(HIDDEN_SIZE);

    for (int h = 0; h < HIDDEN_SIZE; h++) {

        float sum = b1[h];

        for (int i = 0; i < INPUT_SIZE; i++) {
            sum += W1[h * INPUT_SIZE + i] * x[i];
        }

        // ReLU
        hidden[h] = max(0.0f, sum);
    }


    // -------------------------
    // Layer 2: 128 -> 10
    // -------------------------

    vector<float> scores(OUTPUT_SIZE);

    for (int o = 0; o < OUTPUT_SIZE; o++) {

        float sum = b2[o];

        for (int h = 0; h < HIDDEN_SIZE; h++) {
            sum += W2[o * HIDDEN_SIZE + h] * hidden[h];
        }

        scores[o] = sum;
    }


    // -------------------------
    // Find largest score
    // -------------------------

    int prediction = 0;

    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (scores[i] > scores[prediction]) {
            prediction = i;
        }
    }


    // Print scores
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        cout << "Digit " << i
             << " score: " << scores[i] << endl;
    }

    cout << endl;
    cout << "C++ prediction: " << prediction << endl;

    return 0;
}
