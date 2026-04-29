import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results_mpi.csv")

for t in sorted(df["processes"].unique()):
    subset = df[df["processes"] == t]
    plt.plot(subset["size"], subset["time_ms"], label=f"{t} processes")

plt.xlabel("Matrix size")
plt.ylabel("Time (ms)")
plt.title("Matrix multiplication (MPI)")
plt.legend()
plt.grid()

plt.show()