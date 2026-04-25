import numpy as np

def read_matrix(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    M = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        M[i] = list(map(int, lines[i+1].strip().split()))
    return M, n

A, n1 = read_matrix('a.txt')
B, n2 = read_matrix('b.txt')
C_cpp, n3 = read_matrix('c_cpp.txt')

C_numpy = np.dot(A, B)

diff = np.max(np.abs(C_cpp - C_numpy))
print(f"Максимальная разница: {diff}")

if diff == 0:  # для int проверяем точное равенство
    print("Результаты совпадают!")
else:
    print("Результаты различаются!")