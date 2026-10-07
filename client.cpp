#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

using namespace std;

int main() {
    const int PORT = 8080;

    // Blank 28x28 image for our first networking test
    float image[784] = {};

    int sock = socket(AF_INET, SOCK_STREAM, 0);

    sockaddr_in server{};
    server.sin_family = AF_INET;
    server.sin_port = htons(PORT);

    inet_pton(AF_INET, "127.0.0.1", &server.sin_addr);

    if (connect(
        sock,
        reinterpret_cast<sockaddr*>(&server),
        sizeof(server)
    ) < 0) {
        cerr << "Connection failed" << endl;
        return 1;
    }

    send(sock, image, sizeof(image), 0);

    int prediction;

    recv(sock,
         &prediction,
         sizeof(prediction),
         0);

    cout << "Server prediction: "
         << prediction << endl;

    close(sock);

    return 0;
}
