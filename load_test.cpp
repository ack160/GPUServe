#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <numeric>

#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

using namespace std;

const int INPUT_SIZE = 784;
const int PORT = 8080;

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

bool makeRequest(
    const float* image,
    int& prediction,
    double& latencyMs
) {
    auto start = chrono::steady_clock::now();

    int sock = socket(AF_INET, SOCK_STREAM, 0);

    if (sock < 0) {
        return false;
    }

    sockaddr_in server{};
    server.sin_family = AF_INET;
    server.sin_port = htons(PORT);

    inet_pton(
        AF_INET,
        "127.0.0.1",
        &server.sin_addr
    );

    if (connect(
        sock,
        reinterpret_cast<sockaddr*>(&server),
        sizeof(server)
    ) < 0) {
        close(sock);
        return false;
    }

    if (!sendAll(
        sock,
        image,
        INPUT_SIZE * sizeof(float)
    )) {
        close(sock);
        return false;
    }

    if (!recvAll(
        sock,
        &prediction,
        sizeof(prediction)
    )) {
        close(sock);
        return false;
    }

    close(sock);

    auto end = chrono::steady_clock::now();

    latencyMs =
        chrono::duration<double, milli>(
            end - start
        ).count();

    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cout
            << "Usage: ./load_test <concurrency> <requests_per_thread>"
            << endl;

        return 1;
    }

    int concurrency = stoi(argv[1]);
    int requestsPerThread = stoi(argv[2]);

    float image[INPUT_SIZE];

    ifstream file("sample_image.bin", ios::binary);

    if (!file) {
        cerr << "Could not open sample_image.bin" << endl;
        return 1;
    }

    file.read(
        reinterpret_cast<char*>(image),
        sizeof(image)
    );

    if (!file) {
        cerr << "Could not read sample image" << endl;
        return 1;
    }

    vector<double> latencies;
    mutex latencyMutex;

    int successfulRequests = 0;
    int failedRequests = 0;

    mutex countMutex;

    auto testStart = chrono::steady_clock::now();

    vector<thread> threads;

    for (int t = 0; t < concurrency; t++) {

        threads.emplace_back([&]() {

            vector<double> localLatencies;

            int localSuccess = 0;
            int localFailures = 0;

            for (int r = 0; r < requestsPerThread; r++) {

                int prediction;
                double latency;

                if (makeRequest(
                    image,
                    prediction,
                    latency
                )) {
                    localLatencies.push_back(latency);
                    localSuccess++;
                } else {
                    localFailures++;
                }
            }

            {
                lock_guard<mutex> lock(latencyMutex);

                latencies.insert(
                    latencies.end(),
                    localLatencies.begin(),
                    localLatencies.end()
                );
            }

            {
                lock_guard<mutex> lock(countMutex);

                successfulRequests += localSuccess;
                failedRequests += localFailures;
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto testEnd = chrono::steady_clock::now();

    double totalSeconds =
        chrono::duration<double>(
            testEnd - testStart
        ).count();

    if (latencies.empty()) {
        cerr << "No successful requests." << endl;
        return 1;
    }

    sort(latencies.begin(), latencies.end());

    double average =
        accumulate(
            latencies.begin(),
            latencies.end(),
            0.0
        ) / latencies.size();

    size_t p50Index =
        static_cast<size_t>(
            0.50 * (latencies.size() - 1)
        );

    size_t p95Index =
        static_cast<size_t>(
            0.95 * (latencies.size() - 1)
        );
size_t p99Index =
    static_cast<size_t>(
        0.99 * (latencies.size() - 1)
    );
    double throughput =
        successfulRequests / totalSeconds;

    cout << endl;
    cout << "===== GPUServe Load Test =====" << endl;
    cout << "Concurrency:        " << concurrency << endl;
    cout << "Total requests:     "
         << concurrency * requestsPerThread << endl;
    cout << "Successful:         "
         << successfulRequests << endl;
    cout << "Failed:             "
         << failedRequests << endl;
    cout << "Throughput:         "
         << throughput << " req/s" << endl;
    cout << "Average latency:    "
         << average << " ms" << endl;
    cout << "p50 latency:        "
         << latencies[p50Index] << " ms" << endl;
    cout << "p95 latency:        "
         << latencies[p95Index] << " ms" << endl;
cout << "p99 latency:        "
     << latencies[p99Index] << " ms" << endl;
    cout << "==============================" << endl;

    return 0;
}
