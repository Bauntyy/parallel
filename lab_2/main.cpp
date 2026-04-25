#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <string>

using namespace std;

vector<vector<int>> read_matrix(const string& filename, int& n) {
    ifstream f(filename);
    if (!f) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    f >> n;

    vector<vector<int>> M(n, vector<int>(n));

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            f >> M[i][j];

    return M;
}

int main() {

    vector<int> sizes = { 200, 400, 800, 1200, 1600, 2000 };
    vector<int> threads_list = { 1, 2, 4, 8 };

    string path = "data/";

    ofstream fout("results.csv");
    fout << "size,threads,time_ms\n";

    cout << "Size\tThreads\tTime(ms)\n";
    cout << "--------------------------------\n";

    for (int num_threads : threads_list) {

        omp_set_num_threads(num_threads);

        cout << "\n>>> Threads: " << num_threads << "\n";

        for (int n : sizes) {

            int n1, n2;

            auto A = read_matrix(path + "a_" + to_string(n) + ".txt", n1);
            auto B = read_matrix(path + "b_" + to_string(n) + ".txt", n2);

            vector<vector<long long>> C(n, vector<long long>(n, 0));

            auto start = chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {

                    long long sum = 0;

                    for (int k = 0; k < n; k++) {
                        sum += (long long)A[i][k] * B[k][j];
                    }

                    C[i][j] = sum;
                }
            }

            auto end = chrono::high_resolution_clock::now();

            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

            cout << n << "\t" << num_threads << "\t" << duration.count() << endl;

            fout << n << "," << num_threads << "," << duration.count() << "\n";
        }

        cout << "--------------------------------\n";
    }

    fout.close();

    return 0;
}