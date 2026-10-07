#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <algorithm>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>

#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

using namespace std;

const int INPUT_SIZE = 784;
const int MAX_BATCH_SIZE = 8;
const int BATCH_WAIT_MS = 10;

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
        ssize_t n = recv(socket, ptr + received, bytes - received, 0);

        if (n <= 0) {
            return false;
        }

        received += n;
    }

    return true;
}

bool sendAll(int socket, const void* buffer, size_t bytes) {
    const char* ptr = reinterpret_cast<const char*>(buffer);
    size_t sent = 0;

    while (sent < bytes) {
        ssize_t n = send(socket, ptr + sent, bytes - sent, 0);

        if (n <= 0) {
            return false;
        }

        sent += n;
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

struct Request {
    int clientSocket;
    array<float, INPUT_SIZE> image;
};

void batcherLoop(
    queue<Request>& requestQueue,
    mutex& queueMutex,
    condition_variable& queueCV,
    const vector<float>& W1,
    const vector<float>& b1,
    const vector<float>& W2,
    const vector<float>& b2
) {
    while (true) {
        vector<Request> batch;

        {
            unique_lock<mutex> lock(queueMutex);

            queueCV.wait(lock, [&]() {
                return !requestQueue.empty();
            });

            // Take the first request immediately
            batch.push_back(move(requestQueue.front()));
            requestQueue.pop();

            // Give other nearby requests a short chance to join
            auto deadline =
                chrono::steady_clock::now() +
                chrono::milliseconds(BATCH_WAIT_MS);

            while (batch.size() < MAX_BATCH_SIZE) {

                if (!requestQueue.empty()) {
                    batch.push_back(move(requestQueue.front()));
                    requestQueue.pop();
                    continue;
                }

                if (queueCV.wait_until(lock, deadline)
                    == cv_status::timeout) {
                    break;
                }
            }
        }

        vector<int> predictions(batch.size());

        // CPU backend for now.
        // Later this entire batch goes to CUDA together.
        for (size_t i = 0; i < batch.size(); i++) {
            predictions[i] = predict(
                batch[i].image.data(),
                W1,
                b1,
                W2,
                b2
            );
        }

        for (size_t i = 0; i < batch.size(); i++) {
            sendAll(
                batch[i].clientSocket,
                &predictions[i],
                sizeof(predictions[i])
            );

            close(batch[i].clientSocket);
        }

        cout << "Processed batch of "
             << batch.size()
             << " requests"
             << endl;
    }
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

    queue<Request> requestQueue;
    mutex queueMutex;
    condition_variable queueCV;

    thread batcher(
        batcherLoop,
        ref(requestQueue),
        ref(queueMutex),
        ref(queueCV),
        cref(W1),
        cref(b1),
        cref(W2),
        cref(b2)
    );

    batcher.detach();

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

    listen(serverSocket, 64);

    cout << "GPUServe listening on port 8080..." << endl;
    cout << "Max batch size: " << MAX_BATCH_SIZE << endl;

    while (true) {
        int clientSocket = accept(
            serverSocket,
            nullptr,
            nullptr
        );

        if (clientSocket < 0) {
            continue;
        }

        Request request;
        request.clientSocket = clientSocket;

        if (!recvAll(
            clientSocket,
            request.image.data(),
            INPUT_SIZE * sizeof(float)
        )) {
            close(clientSocket);
            continue;
        }

        {
            lock_guard<mutex> lock(queueMutex);
            requestQueue.push(move(request));
        }

        queueCV.notify_one();
    }
}
