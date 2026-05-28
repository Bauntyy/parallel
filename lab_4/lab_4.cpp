#define _CRT_SECURE_NO_WARNINGS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <string> 

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

using namespace std;

const char* kernelSource =
"__kernel void matrixMulKernel(__global const double* A, __global const double* B, __global double* C, int N) {\n"
"    int col = get_global_id(0);\n"
"    int row = get_global_id(1);\n"
"    if (row < N && col < N) {\n"
"        double sum = 0.0;\n"
"        for (int k = 0; k < N; k++) {\n"
"            sum += A[row * N + k] * B[k * N + col];\n"
"        }\n"
"        C[row * N + col] = sum;\n"
"    }\n"
"}\n";

void generateMatrix(const string& filename, int n) {
    ofstream file(filename);
    file << n << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file << rand() % 10;
            if (j < n - 1) file << " ";
        }
        file << endl;
    }
    file.close();
}

vector<vector<double>> readMatrix(const string& filename, int& n) {
    ifstream file(filename);
    file >> n;
    vector<vector<double>> matrix(n, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            file >> matrix[i][j];
    return matrix;
}

void writeResult(const string& filename, const vector<double>& matrix, int n) {
    ofstream file(filename);
    file << n << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file << fixed << setprecision(0) << matrix[i * n + j];
            if (j < n - 1) file << " ";
        }
        file << endl;
    }
}

int main() {
    srand((unsigned)time(nullptr));

    vector<int> sizes = { 200, 400, 800, 1200, 1600, 2000 };
    vector<int> blockSizes = { 8, 16, 32 };
    const int repeats = 3;

    cout << "=====================================================\n";
    cout << "  OpenCL Matrix Multiplication – Automatic Benchmark\n";
    cout << "=====================================================\n\n";

    // Инициализация OpenCL
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;

    clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    cl_int ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);

    if (ret != CL_SUCCESS) {
        cerr << "No GPU device found via OpenCL! Trying CPU fallback...\n";
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, &ret_num_devices);
        if (ret != CL_SUCCESS) {
            cerr << "No OpenCL compatible devices found.\n";
            return 1;
        }
    }

    char deviceName[128];
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    cout << "Using OpenCL Device: " << deviceName << "\n\n";

    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, &ret);
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    if (ret != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        vector<char> log(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        cerr << "Error in kernel compilation:\n" << log.data() << "\n";
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "matrixMulKernel", &ret);

    vector<vector<double>> results(sizes.size(), vector<double>(blockSizes.size(), 0.0));

    for (size_t i = 0; i < sizes.size(); i++) {
        int n = sizes[i];
        cout << "=== Size: " << n << "x" << n << " ===\n";

        string fileA = "matrix_a_" + std::to_string(n) + ".txt";
        string fileB = "matrix_b_" + std::to_string(n) + ".txt";
        generateMatrix(fileA, n);
        generateMatrix(fileB, n);
        cout << "Generated: " << fileA << ", " << fileB << endl;

        int nA, nB;
        auto A = readMatrix(fileA, nA);
        auto B = readMatrix(fileB, nB);
        if (nA != n || nB != n) {
            cerr << "Size mismatch!\n";
            return 1;
        }

        vector<double> h_A(n * n), h_B(n * n), h_C(n * n);
        for (int r = 0; r < n; r++) {
            for (int c = 0; c < n; c++) {
                h_A[r * n + c] = A[r][c];
                h_B[r * n + c] = B[r][c];
            }
        }

        size_t bytes = n * n * sizeof(double);

        cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_A.data(), &ret);
        cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_B.data(), &ret);
        cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &ret);

        for (size_t j = 0; j < blockSizes.size(); j++) {
            size_t bs = blockSizes[j];
            cout << "  Block " << bs << "x" << bs << " : ";

            size_t globalCols = ((n + bs - 1) / bs) * bs;
            size_t globalRows = ((n + bs - 1) / bs) * bs;

            size_t globalWorkSize[2] = { globalCols, globalRows };
            size_t localWorkSize[2] = { bs, bs };

            clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&d_A);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&d_B);
            clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&d_C);
            clSetKernelArg(kernel, 3, sizeof(int), (void*)&n);

            clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
            clFinish(command_queue);

            double totalTime = 0.0;
            for (int r = 0; r < repeats; r++) {
                cl_event event;
                clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &event);
                clWaitForEvents(1, &event);

                cl_ulong time_start, time_end;
                clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
                clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

                double sec = (double)(time_end - time_start) / 1000000000.0;
                totalTime += sec;

                clReleaseEvent(event);
            }

            double avgTime = totalTime / repeats;
            results[i][j] = avgTime;

            long long ops = 2LL * n * n * n;
            double gflops = ops / (avgTime * 1e9);
            cout << fixed << setprecision(4) << avgTime << " sec, " << setprecision(2) << gflops << " GFLOPS\n";

            if (j == blockSizes.size() - 1) {
                clEnqueueReadBuffer(command_queue, d_C, CL_TRUE, 0, bytes, h_C.data(), 0, NULL, NULL);
                string outFile = "matrix_c_" + std::to_string(n) + ".txt";
                writeResult(outFile, h_C, n);
                cout << "    Result saved: " << outFile << "\n";
            }
        }

        clReleaseMemObject(d_A);
        clReleaseMemObject(d_B);
        clReleaseMemObject(d_C);
        cout << "----------------------------------------\n";
    }

    cout << "\n==================== FINAL TABLE ====================\n";
    cout << "Size\\Block |";
    for (int bs : blockSizes) cout << " " << setw(8) << bs << "x" << left << setw(2) << bs << " |";
    cout << "\n------------------------------------------------------\n";
    for (size_t i = 0; i < sizes.size(); i++) {
        cout << setw(8) << sizes[i] << " |";
        for (size_t j = 0; j < blockSizes.size(); j++) {
            cout << " " << setw(10) << fixed << setprecision(4) << results[i][j] << " |";
        }
        cout << "\n";
    }
    cout << "======================================================\n";

    ofstream csv("opencl_benchmark_auto.csv");
    csv << "Size,BlockSize,Time_sec,GFLOPS\n";
    for (size_t i = 0; i < sizes.size(); i++) {
        for (size_t j = 0; j < blockSizes.size(); j++) {
            double gflops = (2LL * sizes[i] * sizes[i] * sizes[i]) / (results[i][j] * 1e9);
            csv << sizes[i] << "," << blockSizes[j] << "," << results[i][j] << "," << gflops << "\n";
        }
    }
    csv.close();
    cout << "\nResults also saved to opencl_benchmark_auto.csv\n";

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    return 0;
}