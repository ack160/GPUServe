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

#include <cuda_runtime.h>

#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

using namespace std;

const int INPUT_SIZE = 784;
const int HIDDEN_SIZE = 128;
const int OUTPUT_SIZE = 10;

const int MAX_BATCH_SIZE = 8;
const int BATCH_WAIT_MS = 10;

// ---------------------------------------------------------
// CUDA error checking
// ---------------------------------------------------------

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        cerr << "CUDA error: "                                \
             << cudaGetErrorString(err)                        \
             << " at line " << __LINE__ << endl;             \
        exit(1);                                               \
    }                                                         \
} while (0)

// ---------------------------------------------------------
// File loading
// ---------------------------------------------------------

template <typename T>
vector<T> loadFile(const string& filename, int count) {
    vector<T> data(count);

    ifstream file(filename, ios::binary);

    if (!file) {
        cerr << "ERROR: Could not open "
             << filename << endl;
        exit(1);
    }

    file.read(
        reinterpret_cast<char*>(data.data()),
        count * sizeof(T)
    );

    if (!file) {
        cerr << "ERROR: Failed reading "
             << filename << endl;
        exit(1);
    }

    return data;
}

// ---------------------------------------------------------
// Networking helpers
// ---------------------------------------------------------

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

        if (n <= 0)
            return false;

        received += n;
    }

    return true;
}

bool sendAll(int socket, const void* buffer, size_t bytes) {
    const char* ptr =
        reinterpret_cast<const char*>(buffer);

    size_t sent = 0;

    while (sent < bytes) {
        ssize_t n = send(
            socket,
            ptr + sent,
            bytes - sent,
            0
        );

        if (n <= 0)
            return false;

        sent += n;
    }

    return true;
}

// ---------------------------------------------------------
// Warp-parallel CUDA kernels
// ---------------------------------------------------------

__global__ void layer1Batch(
    const float* W1,
    const float* b1,
    const float* images,
    float* hidden,
    int batchSize
) {
    int image = blockIdx.x / 16;
    int blockWithinImage = blockIdx.x % 16;

    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    int h = blockWithinImage * 8 + warp;

    if (image >= batchSize)
        return;

    const float* x =
        images + image * INPUT_SIZE;

    const float* weights =
        W1 + h * INPUT_SIZE;

    float sum = 0.0f;

    for (
        int i = lane;
        i < INPUT_SIZE;
        i += 32
    ) {
        sum += weights[i] * x[i];
    }

    for (
        int offset = 16;
        offset > 0;
        offset /= 2
    ) {
        sum += __shfl_down_sync(
            0xffffffff,
            sum,
            offset
        );
    }

    if (lane == 0) {
        sum += b1[h];

        hidden[
            image * HIDDEN_SIZE + h
        ] = sum > 0.0f ? sum : 0.0f;
    }
}

__global__ void layer2Batch(
    const float* W2,
    const float* b2,
    const float* hidden,
    float* scores,
    int batchSize
) {
    int image = blockIdx.x;

    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    if (
        image >= batchSize ||
        warp >= OUTPUT_SIZE
    ) {
        return;
    }

    int o = warp;

    float sum = 0.0f;

    for (
        int h = lane;
        h < HIDDEN_SIZE;
        h += 32
    ) {
        sum +=
            W2[o * HIDDEN_SIZE + h] *
            hidden[
                image * HIDDEN_SIZE + h
            ];
    }

    for (
        int offset = 16;
        offset > 0;
        offset /= 2
    ) {
        sum += __shfl_down_sync(
            0xffffffff,
            sum,
            offset
        );
    }

    if (lane == 0) {
        scores[
            image * OUTPUT_SIZE + o
        ] = sum + b2[o];
    }
}

// ---------------------------------------------------------
// Request
// ---------------------------------------------------------

struct Request {
    int clientSocket;
    array<float, INPUT_SIZE> image;
};

// ---------------------------------------------------------
// GPU inference engine
// ---------------------------------------------------------

class GPUInference {
private:
    float* d_W1 = nullptr;
    float* d_b1 = nullptr;
    float* d_W2 = nullptr;
    float* d_b2 = nullptr;

    struct BufferSlot {
        float* d_images = nullptr;
        float* d_hidden = nullptr;
        float* d_scores = nullptr;

        vector<float> hostImages;
        vector<float> hostScores;
        cudaStream_t stream = nullptr;
    };

    BufferSlot slots[2];

public:
    GPUInference(
        const vector<float>& W1,
        const vector<float>& b1,
        const vector<float>& W2,
        const vector<float>& b2
    ) {
        CUDA_CHECK(cudaMalloc(
            &d_W1,
            W1.size() * sizeof(float)
        ));

        CUDA_CHECK(cudaMalloc(
            &d_b1,
            b1.size() * sizeof(float)
        ));

        CUDA_CHECK(cudaMalloc(
            &d_W2,
            W2.size() * sizeof(float)
        ));

        CUDA_CHECK(cudaMalloc(
            &d_b2,
            b2.size() * sizeof(float)
        ));

        CUDA_CHECK(cudaMemcpy(
            d_W1,
            W1.data(),
            W1.size() * sizeof(float),
            cudaMemcpyHostToDevice
        ));

        CUDA_CHECK(cudaMemcpy(
            d_b1,
            b1.data(),
            b1.size() * sizeof(float),
            cudaMemcpyHostToDevice
        ));

        CUDA_CHECK(cudaMemcpy(
            d_W2,
            W2.data(),
            W2.size() * sizeof(float),
            cudaMemcpyHostToDevice
        ));

        CUDA_CHECK(cudaMemcpy(
            d_b2,
            b2.data(),
            b2.size() * sizeof(float),
            cudaMemcpyHostToDevice
        ));

        for (auto& slot : slots) {
            CUDA_CHECK(cudaStreamCreateWithFlags(
                &slot.stream,
                cudaStreamNonBlocking
            ));

            CUDA_CHECK(cudaMalloc(
                &slot.d_images,
                MAX_BATCH_SIZE *
                INPUT_SIZE *
                sizeof(float)
            ));

            CUDA_CHECK(cudaMalloc(
                &slot.d_hidden,
                MAX_BATCH_SIZE *
                HIDDEN_SIZE *
                sizeof(float)
            ));

            CUDA_CHECK(cudaMalloc(
                &slot.d_scores,
                MAX_BATCH_SIZE *
                OUTPUT_SIZE *
                sizeof(float)
            ));

            slot.hostImages.resize(
                MAX_BATCH_SIZE * INPUT_SIZE
            );

            slot.hostScores.resize(
                MAX_BATCH_SIZE * OUTPUT_SIZE
            );

            CUDA_CHECK(cudaHostRegister(
                slot.hostImages.data(),
                slot.hostImages.size() * sizeof(float),
                cudaHostRegisterDefault
            ));

            CUDA_CHECK(cudaHostRegister(
                slot.hostScores.data(),
                slot.hostScores.size() * sizeof(float),
                cudaHostRegisterDefault
            ));
        }
    }

    vector<int> predictBatch(
        const vector<Request>& batch,
        int slotIndex
    ) {
        int batchSize = batch.size();

        // Each worker has its own independent buffers.
        BufferSlot& slot = slots[slotIndex];

        // Pack requests contiguously
        for (int b = 0; b < batchSize; b++) {
            copy(
                batch[b].image.begin(),
                batch[b].image.end(),
                slot.hostImages.begin() +
                    b * INPUT_SIZE
            );
        }

        CUDA_CHECK(cudaMemcpyAsync(
            slot.d_images,
            slot.hostImages.data(),
            batchSize *
                INPUT_SIZE *
                sizeof(float),
            cudaMemcpyHostToDevice,
            slot.stream
        ));

        layer1Batch<<<
            batchSize * 16,
            256,
            0,
            slot.stream
        >>>(
            d_W1,
            d_b1,
            slot.d_images,
            slot.d_hidden,
            batchSize
        );

        CUDA_CHECK(cudaGetLastError());

        layer2Batch<<<
            batchSize,
            320,
            0,
            slot.stream
        >>>(
            d_W2,
            d_b2,
            slot.d_hidden,
            slot.d_scores,
            batchSize
        );

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(
            slot.hostScores.data(),
            slot.d_scores,
            batchSize *
                OUTPUT_SIZE *
                sizeof(float),
            cudaMemcpyDeviceToHost,
            slot.stream
        ));

        // Wait until all queued GPU operations finish
        // before reading the predictions.
        CUDA_CHECK(cudaStreamSynchronize(slot.stream));

        vector<int> predictions(batchSize);

        for (int b = 0; b < batchSize; b++) {
            int best = 0;

            for (int o = 1; o < OUTPUT_SIZE; o++) {
                if (
                    slot.hostScores[
                        b * OUTPUT_SIZE + o
                    ] >
                    slot.hostScores[
                        b * OUTPUT_SIZE + best
                    ]
                ) {
                    best = o;
                }
            }

            predictions[b] = best;
        }

        return predictions;
    }

    ~GPUInference() {
        cudaFree(d_W1);
        cudaFree(d_b1);
        cudaFree(d_W2);
        cudaFree(d_b2);
        for (auto& slot : slots) {
            cudaStreamSynchronize(slot.stream);
            cudaHostUnregister(slot.hostImages.data());
            cudaHostUnregister(slot.hostScores.data());
            cudaStreamDestroy(slot.stream);
            cudaFree(slot.d_images);
            cudaFree(slot.d_hidden);
            cudaFree(slot.d_scores);
        }
    }
};

// ---------------------------------------------------------
// Dynamic batcher
// ---------------------------------------------------------

void batcherLoop(
    queue<Request>& requestQueue,
    mutex& queueMutex,
    condition_variable& queueCV,
    GPUInference& gpu,
    mutex& assemblyMutex,
    int slotIndex
) {
    while (true) {
        vector<Request> batch;

        {
            // Only one worker assembles a batch at a time.
            lock_guard<mutex> assemblyLock(assemblyMutex);
            unique_lock<mutex> lock(queueMutex);

            queueCV.wait(
                lock,
                [&]() {
                    return !requestQueue.empty();
                }
            );

            batch.push_back(
                move(requestQueue.front())
            );

            requestQueue.pop();

            auto deadline =
                chrono::steady_clock::now() +
                chrono::milliseconds(
                    BATCH_WAIT_MS
                );

            while (
                batch.size() <
                MAX_BATCH_SIZE
            ) {
                if (!requestQueue.empty()) {
                    batch.push_back(
                        move(requestQueue.front())
                    );

                    requestQueue.pop();

                    continue;
                }

                if (
                    queueCV.wait_until(
                        lock,
                        deadline
                    ) == cv_status::timeout
                ) {
                    break;
                }
            }
        }

        vector<int> predictions =
            gpu.predictBatch(batch, slotIndex);

        for (
            size_t i = 0;
            i < batch.size();
            i++
        ) {
            sendAll(
                batch[i].clientSocket,
                &predictions[i],
                sizeof(predictions[i])
            );

            close(
                batch[i].clientSocket
            );
        }
    }
}

// ---------------------------------------------------------
// Main server
// ---------------------------------------------------------

int main() {
    const int PORT = 8080;

    auto W1 = loadFile<float>(
        "model_data/W1.bin",
        HIDDEN_SIZE * INPUT_SIZE
    );

    auto b1 = loadFile<float>(
        "model_data/b1.bin",
        HIDDEN_SIZE
    );

    auto W2 = loadFile<float>(
        "model_data/W2.bin",
        OUTPUT_SIZE * HIDDEN_SIZE
    );

    auto b2 = loadFile<float>(
        "model_data/b2.bin",
        OUTPUT_SIZE
    );

    GPUInference gpu(
        W1,
        b1,
        W2,
        b2
    );

    queue<Request> requestQueue;
    mutex queueMutex;
    condition_variable queueCV;
    mutex assemblyMutex;

    thread batcher0(
        batcherLoop,
        ref(requestQueue),
        ref(queueMutex),
        ref(queueCV),
        ref(gpu),
        ref(assemblyMutex),
        0
    );



    batcher0.detach();


    int serverSocket =
        socket(AF_INET, SOCK_STREAM, 0);

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

    if (
        bind(
            serverSocket,
            reinterpret_cast<sockaddr*>(
                &address
            ),
            sizeof(address)
        ) < 0
    ) {
        cerr << "Bind failed" << endl;
        return 1;
    }

    listen(serverSocket, 64);

    cout << "GPUServe GPU server listening on port 8080..." << endl;
    cout << "Max batch size: "
         << MAX_BATCH_SIZE << endl;

    while (true) {
        int clientSocket =
            accept(
                serverSocket,
                nullptr,
                nullptr
            );

        if (clientSocket < 0)
            continue;

        Request request;
        request.clientSocket =
            clientSocket;

        if (
            !recvAll(
                clientSocket,
                request.image.data(),
                INPUT_SIZE *
                sizeof(float)
            )
        ) {
            close(clientSocket);
            continue;
        }

        {
            lock_guard<mutex>
                lock(queueMutex);

            requestQueue.push(
                move(request)
            );
        }

        queueCV.notify_one();
    }
}
