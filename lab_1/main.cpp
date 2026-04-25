#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

using namespace std;

int main() {
    ifstream fa("a.txt"), fb("b.txt");
    int n;
    fa >> n;
    fb >> n;
    
    vector<vector<int>> A(n, vector<int>(n));
    vector<vector<int>> B(n, vector<int>(n)); 
    vector<vector<long long>> C(n, vector<long long>(n, 0));
    
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            fa >> A[i][j];
            
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            fb >> B[i][j];
    
    auto start = chrono::high_resolution_clock::now();
    
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            for(int k = 0; k < n; k++)
                C[i][j] += (long long)A[i][k] * B[k][j]; 
    
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    ofstream fc("c_cpp.txt");
    fc << n << endl;
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++)
            fc << C[i][j] << " ";
        fc << endl;
    }
    
    cout << "C++: " << n << "x" << n << " time: " << duration.count() << " ms\n";
    
    return 0;
}