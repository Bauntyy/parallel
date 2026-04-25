import numpy as np
import os

sizes = [100, 200, 400, 800, 1200, 1600, 2000]

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")
os.makedirs(data_dir, exist_ok=True)

def generate_matrix(n, filename):
    M = np.random.randint(0, 100, (n, n))
    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        for i in range(n):
            f.write(' '.join(map(str, M[i])) + '\n')

for n in sizes:
    generate_matrix(n, os.path.join(data_dir, f"a_{n}.txt"))
    generate_matrix(n, os.path.join(data_dir, f"b_{n}.txt"))
    print(f"Сгенерированы {n}x{n}")

print("Готово!")