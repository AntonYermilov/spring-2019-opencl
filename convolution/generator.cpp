#include <iostream>
#include <cassert>
using std::cout;

int main(int argc, const char ** argv) {
    assert(argc == 3);
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);

    cout << n << ' ' << m << '\n';
    for (int i = 0; i != n; ++i) {
        for (int j = 0; j != n; ++j)
            cout << 1 << ' ';
        cout << '\n';
    }
    for (int i = 0; i != m; ++i) {
        for (int j = 0; j != m; ++j)
            cout << 1 << ' ';
        cout << '\n';
    }

    return 0;
}
