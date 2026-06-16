import re
import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt

# Read output logs
with open("output.log", "r") as file:
    log_data = file.read()

# Define matrix sizes in output logs
matrix_sizes = [10, 100, 1000, 2000, 3000,
                4000, 5000, 6000, 7000, 8000, 9000, 10000]

agg_cublas_times = []
agg_cuda_times = []
agg_blas_times = []
agg_openmp_times = []
agg_cpp_threads_times = []

agg_cublas_stats = []
agg_cuda_stats = []
agg_blas_stats = []
agg_openmp_stats = []
agg_cpp_threads_stats = []


def compile_gpu(matrix_size):
    """
    @brief Create regex pattern for gpu output
    @param matrix_size Size of matrix
    @return Compiled regex
    """
    pattern = re.compile(
        rf"Matrix size = {matrix_size}, Block size = 32\n"
        r"CuBLAS execution elapsed time \(ms\) = ([\d.]+)\n"
        r"CuBLAS copy elapsed time \(ms\) = [\d.]+\n"
        r"CUDA execution elapsed time \(ms\) = ([\d.]+)",
        re.MULTILINE
    )
    return pattern


def extract_gpu():
    """
    @brief Extract data from gpu output
    """
    for size in matrix_sizes:
        pattern = compile_gpu(size)
        cublas_times = []
        cuda_times = []
        for match in pattern.finditer(log_data):
            cublas_times.append(float(match.group(1)))
            cuda_times.append(float(match.group(2)))
        agg_cublas_times.append(cublas_times)
        agg_cuda_times.append(cuda_times)


def compile_cpu(matrix_size):
    """
    @brief Create regex pattern for cpu output
    @param matrix_size Size of matrix
    @return Compiled regex
    """
    pattern = re.compile(
        rf"Matrix size = {matrix_size}, Block size = 32\n"
        r"BLAS elapsed time \(ms\) = ([\d.]+)\n"
        r"OpenMP elapsed time \(ms\) = ([\d.]+)\n"
        r"C\+\+ threads elapsed time \(ms\) = ([\d.]+)",
        re.MULTILINE
    )
    return pattern


def extract_cpu():
    """
    @brief Extract data from cpu output
    """
    for size in matrix_sizes:
        pattern = compile_cpu(size)
        blas_times = []
        openmp_times = []
        cpp_threads_times = []
        for match in pattern.finditer(log_data):
            blas_times.append(float(match.group(1)))
            openmp_times.append(float(match.group(2)))
            cpp_threads_times.append(float(match.group(3)))
        agg_blas_times.append(blas_times)
        agg_openmp_times.append(openmp_times)
        agg_cpp_threads_times.append(cpp_threads_times)


def bootstrap(data):
    """
    @brief Statistical bootstrap method
    @param data 1D array of runtimes
    @return Sample mean with confidence intervals and variance
    """
    n_bootstrap = 10_000
    bootstrap_means = np.mean(np.random.choice(
        data, size=(n_bootstrap, len(data)), replace=True), axis=1)
    mean = np.mean(bootstrap_means)
    low = np.percentile(bootstrap_means, 2.5)
    high = np.percentile(bootstrap_means, 97.5)
    return mean, low, high, bootstrap_means


def flop(x):
    """
    @brief Calculate floating point operations for square matrix multiplication
    @param x Matrix size
    @return Floating point operations count
    """
    return 2 * x ** 3 - x ** 2


def sec(x):
    """
    @brief Convert milliseconds to seconds
    @param x Milliseconds
    @return Seconds
    """
    return x * 1000


def flops(n, ms):
    """
    @brief Calculate floating point operations per second
    @param n Matrix size
    @param ms Milliseconds
    @return Floating point operations per second
    """
    return flop(n) / sec(ms)


def transform():
    """
    @brief Convert runtimes to statistics
    """
    for it in range(len(matrix_sizes)):
        agg_cublas_stats.append((lambda m, l, h, b: (m, l, h, flops(matrix_sizes[it], m), flops(matrix_sizes[it], l), flops(matrix_sizes[it], h), b))(
            *bootstrap(agg_cublas_times[it])))
        agg_cuda_stats.append((lambda m, l, h, b: (m, l, h, flops(matrix_sizes[it], m), flops(matrix_sizes[it], l), flops(matrix_sizes[it], h), b))(
            *bootstrap(agg_cuda_times[it])))
        agg_blas_stats.append((lambda m, l, h, b: (m, l, h, flops(matrix_sizes[it], m), flops(matrix_sizes[it], l), flops(matrix_sizes[it], h), b))(
            *bootstrap(agg_blas_times[it])))
        agg_openmp_stats.append((lambda m, l, h, b: (m, l, h, flops(matrix_sizes[it], m), flops(matrix_sizes[it], l), flops(matrix_sizes[it], h), b))(
            *bootstrap(agg_openmp_times[it])))
        agg_cpp_threads_stats.append((lambda m, l, h, b: (m, l, h, flops(matrix_sizes[it], m), flops(matrix_sizes[it], l), flops(matrix_sizes[it], h), b))(
            *bootstrap(agg_cpp_threads_times[it])))


def load_plots():
    """
    @brief Generate plots from runtime statistics
    """

    # Initialize data
    cublas_arr = np.array([stats[:-1] for stats in agg_cublas_stats])
    cuda_arr = np.array([stats[:-1] for stats in agg_cuda_stats])
    blas_arr = np.array([stats[:-1] for stats in agg_blas_stats])
    openmp_arr = np.array([stats[:-1] for stats in agg_openmp_stats])
    cpp_arr = np.array([stats[:-1] for stats in agg_cpp_threads_stats])
    means, lows, highs = [], [], []

    # Collect CuBLAS data
    cublas_means = cublas_arr[2:, 3].tolist()
    cublas_lows = cublas_arr[2:, 4].tolist()
    cublas_highs = cublas_arr[2:, 5].tolist()
    means.append(cublas_means), lows.append(
        cublas_lows), highs.append(cublas_highs)

    # Collect CUDA data
    cuda_means = cuda_arr[2:, 3].tolist()
    cuda_lows = cuda_arr[2:, 4].tolist()
    cuda_highs = cuda_arr[2:, 5].tolist()
    means.append(cuda_means), lows.append(cuda_lows), highs.append(cuda_highs)

    # Collect BLAS data
    blas_means = blas_arr[2:, 3].tolist()
    blas_lows = blas_arr[2:, 4].tolist()
    blas_highs = blas_arr[2:, 5].tolist()
    means.append(blas_means), lows.append(blas_lows), highs.append(blas_highs)

    # Collect OpenMP data
    openmp_means = openmp_arr[2:, 3].tolist()
    openmp_lows = openmp_arr[2:, 4].tolist()
    openmp_highs = openmp_arr[2:, 5].tolist()
    means.append(openmp_means), lows.append(
        openmp_lows), highs.append(openmp_highs)

    # Collect C++ Threads data
    cpp_means = cpp_arr[2:, 3].tolist()
    cpp_lows = cpp_arr[2:, 4].tolist()
    cpp_highs = cpp_arr[2:, 5].tolist()
    means.append(cpp_means), lows.append(cpp_lows), highs.append(cpp_highs)

    # Plot data
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    labels = ["CuBLAS", "CUDA", "BLAS", "OpenMP", "C++ Threads"]
    for it in range(0, 5):
        plt.plot(matrix_sizes[2:], means[it],
                 color=colors[it], label=labels[it])
        plt.fill_between(
            matrix_sizes[2:], lows[it], highs[it], color=colors[it], alpha=0.2)

    # Display plot
    plt.xlabel("Matrix Sizes (N)")
    plt.ylabel("Performance (FLOPS)")
    plt.title('Performance v.s. Matrix Size with Confidence Intervals')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def load_tables():
    """
    @brief Generate table of runtime statistics
    """

    # Initialize data
    cublas_arr = np.array([stats[:-1] for stats in agg_cublas_stats])
    cuda_arr = np.array([stats[:-1] for stats in agg_cuda_stats])
    blas_arr = np.array([stats[:-1] for stats in agg_blas_stats])
    openmp_arr = np.array([stats[:-1] for stats in agg_openmp_stats])
    cpp_arr = np.array([stats[:-1] for stats in agg_cpp_threads_stats])
    arrs = [cublas_arr, cuda_arr, blas_arr, openmp_arr, cpp_arr]
    matrix_sizes = list(range(1000, 11000, 1000))
    algorithms = ["CuBLAS", "CUDA", "BLAS", "OpenMP", "C++ Threads"]

    # Define rows for table
    rows = []
    for i, size in enumerate(matrix_sizes):
        for j, alg in enumerate(algorithms):
            if j == 0:
                size_label = size
            else:
                size_label = ""
            rows.append([
                size_label,
                alg,
                f"{arrs[j][2+i, 3]:.2e}",
                f"{arrs[j][2+i, 4]:.2e}",
                f"{arrs[j][2+i, 5]:.2e}",
            ])

    # Create table
    rows_array = np.array(rows)
    columns = ["Matrix Size", "Algorithm",
               "Mean FLOPS", "Best FLOPS", "Worst FLOPS"]
    _, ax = plt.subplots(figsize=(11, 13))
    ax.axis('off')
    ax.set_title("Algorithm Performance by Matrix Size", fontsize=14, pad=16)
    table = ax.table(
        cellText=rows_array,
        colLabels=columns,
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    # Color alternating cell groups
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            continue
        group_number = (row - 1) // 5
        if group_number % 2 == 0:
            cell.set_facecolor("#ffffff")
        else:
            cell.set_facecolor("#d0d0d0")

    # Display table
    plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
    plt.savefig("table.png", dpi=300, bbox_inches="tight")


def load_std_devs():
    """
    @brief Find standard deviations of performances from largest matrix size
    """
    matrix_size = 10_000
    cublas_flops = flops(matrix_size, agg_cublas_stats[-1][6])
    cuda_flops = flops(matrix_size, agg_cuda_stats[-1][6])
    blas_flops = flops(matrix_size, agg_blas_stats[-1][6])
    openmp_flops = flops(matrix_size, agg_openmp_stats[-1][6])
    cpp_flops = flops(matrix_size, agg_cpp_threads_stats[-1][6])
    print(f"{np.sqrt(np.var(cublas_flops, ddof=1)):.3f}")
    print(f"{np.sqrt(np.var(cuda_flops, ddof=1)):.3f}")
    print(f"{np.sqrt(np.var(blas_flops, ddof=1)):.3f}")
    print(f"{np.sqrt(np.var(openmp_flops, ddof=1)):.3f}")
    print(f"{np.sqrt(np.var(cpp_flops, ddof=1)):.3f}")


def load_significance():
    """
    @brief Perform formal test to find statistical significance
    """

    # Initialize data
    matrix_size = 10_000
    n_samples = 10_000
    cublas = flops(matrix_size, agg_cublas_stats[-1][6])
    cuda = flops(matrix_size, agg_cuda_stats[-1][6])
    blas = flops(matrix_size, agg_blas_stats[-1][6])
    openmp = flops(matrix_size, agg_openmp_stats[-1][6])
    cpp = flops(matrix_size, agg_cpp_threads_stats[-1][6])

    # Create dataframe
    values = np.concatenate([cublas, cuda, blas, openmp, cpp])
    groups = (
        ["cublas"] * n_samples
        + ["cuda"] * n_samples
        + ["blas"] * n_samples
        + ["openmp"] * n_samples
        + ["cpp"] * n_samples
    )
    df = pd.DataFrame({"value": values, "group": groups})

    # Perform Welch ANOVA test
    welch_anova = pg.welch_anova(dv="value", between="group", data=df)
    print(welch_anova)

    # Perform Games-Howell test
    posthoc = pg.pairwise_gameshowell(dv="value", between="group", data=df)
    print(posthoc)


if __name__ == "__main__":
    extract_gpu()
    extract_cpu()
    transform()
    # load_plots()
    # load_tables()
    # load_std_devs()
    # load_significance()
