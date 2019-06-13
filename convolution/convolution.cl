__kernel void apply_convolution(int N, int M, __global const float * A, __global const float * B, __global float * C) {
    int id = get_global_id(0);
    if (id >= N * N)
        return;

    int r0 = id / N, c0 = id % N, k = M / 2;
    for (int i = -k; i <= k; ++i) {
        for (int j = -k; j <= k; ++j) {
            int rA = r0 + i, cA = c0 + j;
            if (0 <= rA  && rA < N && 0 <= cA && cA < N) {
                int rB = i + k, cB = j + k;
                C[id] += A[rA * N + cA] * B[rB * M + cB];
            }
        }
    }
}
