#include <fstream>
#include <cassert>
#include <cmath>
#include <vector>

using std::ifstream;
using std::vector;

int main() {
    ifstream in("input.txt");

    int n;
    in >> n;

    vector<double> data(n);
    for (int i = 0; i < n; ++i)
        in >> data[i];

    in.close();

    vector<double> sum(n);
    for (int i = 1; i < n; ++i)
        sum[i] = sum[i - 1] + data[i - 1];

    in = ifstream("output.txt");

    for (int i = 0; i < n; ++i) {
        double x;
        in >> x;
        assert(fabsl(x - sum[i]) < 1e-6);
    }

    in.close();

    return 0;
}
