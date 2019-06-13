#include <iostream>
#include <cassert>

using std::cout;

int main(int argc, const char ** argv) {
    assert(argc == 2);
    int n = atoi(argv[1]);

    cout << n << '\n';
    for (int i = 0; i != n; ++i) {
        cout << 0.5 + 0.5 * (rand() % 3) << ' ';
    }
    cout << '\n';

    return 0;
}
