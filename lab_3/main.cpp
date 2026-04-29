#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <string>

using namespace std;

// чтение матрицы (как у тебя было)
vector<int> read_matrix(const string& filename, int& n) {
    ifstream f(filename);
    if (!f) {
        cerr << "Ошибка открытия файла: " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    f >> n;
    vector<int> M(n * n);

    for (int i = 0; i < n * n; i++)
        f >> M[i];

    return M;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<int> sizes = { 200, 400, 800, 1200, 1600, 2000 };

    ofstream fout;
    if (rank == 0) {
        fout.open("results_mpi.csv", ios::app);
        if (fout.tellp() == 0)
            fout << "size,processes,time_ms\n";
    }

    for (int n : sizes) {

        vector<int> A, B, C;

        int n1 = 0, n2 = 0;

        // читаем ТОЛЬКО в rank 0
        if (rank == 0) {
            string path = "data/";
            A = read_matrix(path + "a_" + to_string(n) + ".txt", n1);
            B = read_matrix(path + "b_" + to_string(n) + ".txt", n2);

            if (n1 != n || n2 != n) {
                cerr << "Размеры матриц не совпадают!" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            C.resize(n * n);
        }

        // всем нужен размер n
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // всем нужна матрица B
        if (rank != 0)
            B.resize(n * n);

        MPI_Bcast(B.data(), n * n, MPI_INT, 0, MPI_COMM_WORLD);

        int rows = n / size;

        vector<int> local_A(rows * n);
        vector<int> local_C(rows * n, 0);

        // раздаём A
        MPI_Scatter(A.data(), rows * n, MPI_INT,
            local_A.data(), rows * n, MPI_INT,
            0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);

        auto start = chrono::high_resolution_clock::now();

        // умножение
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < n; j++) {
                int sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += local_A[i * n + k] * B[k * n + j];
                }
                local_C[i * n + j] = sum;
            }
        }

        auto end = chrono::high_resolution_clock::now();

        // собираем результат
        MPI_Gather(local_C.data(), rows * n, MPI_INT,
            C.data(), rows * n, MPI_INT,
            0, MPI_COMM_WORLD);

        if (rank == 0) {
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

            cout << "Size " << n
                << " | Proc " << size
                << " | Time " << duration.count() << " ms\n";

            fout << n << "," << size << "," << duration.count() << "\n";
        }
    }

    if (rank == 0) fout.close();

    MPI_Finalize();
    return 0;
}