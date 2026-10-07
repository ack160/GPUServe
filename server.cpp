#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <thread>

#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

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

bool recvAll(int socket, void* buffer, size_t bytes) {
    char* ptr = reinterpret_cast<char*>(buffer);
    size_t received = 0;

    while (received < bytes) {
        ssize_t n = recv(
            socket,
            ptr + received,
            bytes - received,
            0
        );

        if (n <= 0) {
            return false;
        }

        received += n;
    }

    return true;
}

int predict(
    const float* x,
    const vector<float>& W1,
    const vector<float>& b1,
    const vector<float>& W2,
    const vector<float>& b2
) {
    vector<float> hidden(128);

    for (int h = 0; h < 128; h++) {
        float sum = b1[h];

        for (int i = 0; i < 784; i++) {
            sum += W1[h * 784 + i] * x[i];
        }

        hidden[h] = max(0.0f, sum);
    }

    float scores[10];

    for (int o = 0; o < 10; o++) {
        float sum = b2[o];

        for (int h = 0; h < 128; h++) {
            sum += W2[o * 128 + h] * hidden[h];
        }

        scores[o] = sum;
    }

    int prediction = 0;

    for (int i = 1; i < 10; i++) {
        if (scores[i] > scores[prediction]) {
            prediction = i;
        }
    }

    return prediction;
}

void handleClient(
    int clientSocket,
    const vector<float>& W1,
    const vector<float>& b1,
    const vector<float>& W2,
    const vector<float>& b2
) {
    float image[784];

    if (recvAll(clientSocket, image, sizeof(image))) {
        int prediction = predict(
            image,
            W1,
            b1,
            W2,
            b2
        );

        send(
            clientSocket,
            &prediction,
            sizeof(prediction),
            0
        );

        cout << "Prediction: "
             << prediction
             << endl;
    }

    close(clientSocket);
}

int main() {
    const int PORT = 8080;

    auto W1 =
        loadFile<float>("model_data/W1.bin", 128 * 784);

    auto b1 =
        loadFile<float>("model_data/b1.bin", 128);

    auto W2 =
        loadFile<float>("model_data/W2.bin", 10 * 128);

    auto b2 =
        loadFile<float>("model_data/b2.bin", 10);

    int serverSocket = socket(AF_INET, SOCK_STREAM, 0);

    int opt = 1;
    setsockopt(
        serverSocket,
        SOL_SOCKET,
        SO_REUSEADDR,
        &opt,
        sizeof(opt)
    );

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(
        serverSocket,
        reinterpret_cast<sockaddr*>(&address),
        sizeof(address)
    ) < 0) {
        cerr << "Bind failed" << endl;
        return 1;
    }

    listen(serverSocket, 16);

    cout << "GPUServe listening on port 8080..." << endl;

    while (true) {
        int clientSocket = accept(
            serverSocket,
            nullptr,
            nullptr
        );

        if (clientSocket < 0) {
            continue;
        }

        thread worker(
            handleClient,
            clientSocket,
            cref(W1),
            cref(b1),
            cref(W2),
            cref(b2)
        );

        worker.detach();
    }
}
