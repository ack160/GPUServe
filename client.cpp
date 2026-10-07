#include <iostream>
#include <fstream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

using namespace std;

int main() {
    const int PORT = 8080;

    float image[784];

    ifstream file("sample_image.bin", ios::binary);

    if (!file) {
        cerr << "Could not open sample_image.bin" << endl;
        return 1;
    }

    file.read(
        reinterpret_cast<char*>(image),
        sizeof(image)
    );

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

    recv(sock, &prediction, sizeof(prediction), 0);

    cout << "Server prediction: " << prediction << endl;

    close(sock);
    return 0;
}
