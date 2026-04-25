import numpy as np

def generate_matrix(n, filename):
    M = np.random.randint(0, 100, (n, n))
    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        for i in range(n):
            f.write(' '.join(map(str, M[i])) + '\n')
    return M

n = 1200
generate_matrix(n, 'a.txt')
generate_matrix(n, 'b.txt')
print(f"Сгенерированы матрицы {n}x{n} в файлы a.txt и b.txt")