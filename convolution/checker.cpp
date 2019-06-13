#include <iostream>
#include <cassert>
#include <cmath>

inline bool inside(int i, int j, int n) {
    return 0 <= i && i < n && 0 <= j && j < n;
}

int main(int argc, const char ** argv) {
    assert(argc == 3);
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);

    for (int i = 0; i != n; ++i) {
        for (int j = 0; j != n; ++j) {
            int k = m / 2;
            int sum = 0;
            for (int di = -k; di <= k; ++di) {
                for (int dj = -k; dj <= k; ++dj) {
                    sum += inside(i + di, j + dj, n);
                }
            }

            float x;
            std::cin >> x;
            assert(fabs(sum - x) < 1e-3);
        }
    }
}
