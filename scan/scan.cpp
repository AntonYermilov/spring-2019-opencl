#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>

#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"

using std::cin;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;

using std::vector;

const size_t WG_SIZE = 256;
typedef float Double;

vector<Double> read_data(const std::string& input_file) {
    std::ifstream in(input_file);
    
    int n;
    in >> n;

    vector<Double> a(n);
    for (int i = 0; i != n; ++i)
        in >> a[i];

    in.close();
    return a;
}

void write_data(const std::string& output_file, const vector<Double>& a) {
    std::ofstream out(output_file);
    out.precision(10);
    out << fixed;

    for (auto& x : a)
        out << x << ' ';
    out << endl;

    out.close();
}

cl::Context context;
cl::Program program;
cl::CommandQueue queue;

cl_ulong total_time = 0;

void scan_k(vector<Double>& a) {
    int size = a.size();
    int k = (size + 2 * WG_SIZE - 1) / (2 * WG_SIZE);

    cl::Buffer input(context, CL_MEM_READ_ONLY, sizeof(Double) * size);
    cl::Buffer output(context, CL_MEM_READ_WRITE, sizeof(Double) * size);

    queue.enqueueWriteBuffer(input, CL_TRUE, 0, sizeof(Double) * size, &a[0]);

    cl::Kernel bs(program, "blelloch_scan");
    bs.setArg(0, cl_int(size));
    bs.setArg(1, input);
    bs.setArg(2, output);
    bs.setArg(3, cl::__local(sizeof(Double) * (2 * WG_SIZE)));
    
    cl::Event event;
    queue.enqueueNDRangeKernel(bs, cl::NullRange, cl::NDRange(k * WG_SIZE), cl::NDRange(WG_SIZE), NULL, &event);
    event.wait();

    cl_ulong start_time = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end_time = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    cl_ulong elapsed_time = end_time - start_time;
    total_time += elapsed_time;

    queue.enqueueReadBuffer(output, CL_TRUE, 0, sizeof(Double) * size, &a[0]);
}

vector<Double> get_parts(vector<Double>& a) {
    int size = a.size();
    
    int k = (size + 2 * WG_SIZE - 1) / (2 * WG_SIZE);
    int kr = ((k + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;

    vector<Double> parts(k, 0);

    cl::Buffer input(context, CL_MEM_READ_ONLY, sizeof(Double) * size);
    cl::Buffer partials(context, CL_MEM_WRITE_ONLY, sizeof(Double) * k);

    queue.enqueueWriteBuffer(input, CL_TRUE, 0, sizeof(Double) * size, &a[0]);

    cl::Kernel bs(program, "get_parts");
    bs.setArg(0, cl_int(size));
    bs.setArg(1, input);
    bs.setArg(2, partials);
    
    cl::Event event;
    queue.enqueueNDRangeKernel(bs, cl::NullRange, cl::NDRange(kr), cl::NDRange(WG_SIZE), NULL, &event);
    event.wait();

    cl_ulong start_time = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end_time = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    cl_ulong elapsed_time = end_time - start_time;
    total_time += elapsed_time;

    queue.enqueueReadBuffer(partials, CL_TRUE, 0, sizeof(Double) * k, &parts[0]);

    return parts;
}

void apply_parts(vector<Double>& a, vector<Double>& parts) {
    int size = a.size();
    int k = (size + 2 * WG_SIZE - 1) / (2 * WG_SIZE);

    cl::Buffer input(context, CL_MEM_READ_WRITE, sizeof(Double) * size);
    cl::Buffer partials(context, CL_MEM_READ_ONLY, sizeof(Double) * k);

    queue.enqueueWriteBuffer(input, CL_TRUE, 0, sizeof(Double) * size, &a[0]);
    queue.enqueueWriteBuffer(partials, CL_TRUE, 0, sizeof(Double) * k, &parts[0]);

    cl::Kernel bs(program, "apply_parts");
    bs.setArg(0, cl_int(size));
    bs.setArg(1, input);
    bs.setArg(2, partials);
    
    cl::Event event;
    queue.enqueueNDRangeKernel(bs, cl::NullRange, cl::NDRange(k * WG_SIZE), cl::NDRange(WG_SIZE), NULL, &event);
    event.wait();

    cl_ulong start_time = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end_time = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    cl_ulong elapsed_time = end_time - start_time;
    total_time += elapsed_time;

    queue.enqueueReadBuffer(input, CL_TRUE, 0, sizeof(Double) * size, &a[0]);
}

void print(vector<Double>& a, char end=0) {
    for (auto x : a)
        cout << x << ' ';
    cout << '\n';
    if (end)
        cout << end;
}

void scan(vector<Double>& a) {
    scan_k(a);
    // print(a);
    auto p = get_parts(a);
    // print(p);
    if (a.size() > 2 * WG_SIZE) {
        // cout << ">\n";
        scan(p);
        // cout << "<\n";
    }
    apply_parts(a, p);
    // print(a);
}

int main() {
    cout.precision(10);
    cout << fixed << '\n';

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
    
    context = cl::Context({default_device});

    cl::Program::Sources sources;

    vector<std::string> cl_files = {"scan.cl"};
    vector<std::string> cl_strings;
    for (auto& cl_file : cl_files) {
        std::ifstream cl_ifstream(cl_file);
        std::string cl_string(std::istreambuf_iterator<char>(cl_ifstream), (std::istreambuf_iterator<char>()));
        cl_strings.emplace_back(cl_string);
        sources.push_back({cl_strings.back().c_str(), cl_strings.back().length()});
    }

    try {
        program = cl::Program(context, sources);
        if (program.build() != CL_SUCCESS) {
            cout << "An error occurred while building program: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
            exit(1);
        }
        queue = cl::CommandQueue(context, default_device, CL_QUEUE_PROFILING_ENABLE);
        
        vector<Double> a = read_data("input.txt");
        scan(a);
        write_data("output.txt", a);
    } catch (cl::Error e) {
        cerr << endl << e.what() << " : " << e.err() << endl;
        exit(1);
    }
    
    cout << "Total time: " << total_time / 1000000.0 << " ms" << endl;

    return 0;
}
