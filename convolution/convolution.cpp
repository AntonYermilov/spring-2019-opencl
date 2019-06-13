#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>

#include "cl.hpp"

using std::cin;
using std::cout;
using std::cerr;
using std::fixed;

using std::vector;

int N, M;
vector<float> A, B, C;

void read_data(const std::string& input_file) {
    std::ifstream in(input_file);
    
    in >> N >> M;

    A.resize(N * N);
    B.resize(M * M);
    C.resize(N * N);
    
    for (int i = 0; i != N * N; ++i)
        in >> A[i];
    
    for (int i = 0; i != M * M; ++i)
        in >> B[i];

    in.close();
}

void write_data(const std::string& output_file) {
    std::ofstream out(output_file);
    out.precision(3);
    out << fixed;

    for (int i = 0; i != N; ++i) {
        for (int j = 0; j != N; ++j)
            out << C[i * N + j] << ' ';
        out << '\n';
    }

    out.flush();
    out.close();
}

int main(int argc, const char ** argv) {
    vector<cl::Platform> platforms;  
    cl::Platform::get(&platforms);
    assert(platforms.size() > 0);
    cl::Platform default_platform = platforms[0];

    vector<cl::Device> devices;
    default_platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    assert(devices.size() > 0);
    cl::Device default_device = devices[0];

#ifdef LOCAL
    cerr << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
    cerr << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";
#endif
    
    cl::Context context({default_device});
    cl::Program::Sources sources;

    vector<std::string> cl_files = {"convolution.cl"};
    vector<std::string> cl_strings;
    for (auto& cl_file : cl_files) {
        std::ifstream cl_ifstream(cl_file);
        std::string cl_string(std::istreambuf_iterator<char>(cl_ifstream), (std::istreambuf_iterator<char>()));
        cl_strings.emplace_back(cl_string);
        sources.push_back({cl_strings.back().c_str(), cl_strings.back().length()});
    }

    cl::Program program(context, sources);
    if (program.build() != CL_SUCCESS) {
        cout << "An error occurred while building program: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
        exit(1);
    }

    read_data("input.txt");
    
    cl::Buffer buf_A(context, CL_MEM_READ_ONLY, sizeof(float) * A.size());
    cl::Buffer buf_B(context, CL_MEM_READ_ONLY, sizeof(float) * B.size());
    cl::Buffer buf_C(context, CL_MEM_READ_WRITE, sizeof(float) * C.size());

    cl::CommandQueue queue(context, default_device);
    queue.enqueueWriteBuffer(buf_A, CL_TRUE, 0, sizeof(float) * A.size(), &A[0]);
    queue.enqueueWriteBuffer(buf_B, CL_TRUE, 0, sizeof(float) * B.size(), &B[0]);

    cl::Kernel kernel(program, "apply_convolution");
    kernel.setArg(0, cl_int(N));
    kernel.setArg(1, cl_int(M));
    kernel.setArg(2, buf_A);
    kernel.setArg(3, buf_B);
    kernel.setArg(4, buf_C);

    // queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N * N), cl::NullRange);

#ifdef LOCAL
    int cur_time = clock();
#endif

    int block_size = argc == 1 ? 16 : atoi(argv[1]);
    int number_of_blocks = (N * N + block_size - 1) / block_size;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(block_size * number_of_blocks), cl::NDRange(block_size));
    queue.finish();

#ifdef LOCAL
    cur_time = clock() - cur_time;
    cerr << "\n" << 1.0 * cur_time / CLOCKS_PER_SEC << " sec.\n";
#endif

    queue.enqueueReadBuffer(buf_C, CL_TRUE, 0, sizeof(float) * C.size(), &C[0]);
    write_data("output.txt");

    return 0;
}
