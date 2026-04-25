import matplotlib.pyplot as plt

# Твои данные (скопируй сюда свои размеры и время)
sizes = [100, 200, 400, 600, 800, 1200, 1600]
times = [11, 89, 721,2431, 5935, 21233, 54762]

# Построение графика
plt.plot(sizes, times, 'b-o')  # b - синий, o - точки
plt.xlabel('Размер матрицы N')
plt.ylabel('Время (мс)')
plt.title('Время умножения матриц')
plt.grid(True)
plt.show()