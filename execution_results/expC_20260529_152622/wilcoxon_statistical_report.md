# Global Statistical Analysis Report (30 Runs)

This report statistically analyzes the performance of Decomposition mode vs Traditional mode.

## Performance Metrics (Mean, Median, Standard Deviation and Wilcoxon)

| Metric | Mean Decomp. | Median Decomp. | Std. Dev. Decomp. | Mean Trad. | Median Trad. | Std. Dev. Trad. | p-value (Wilcoxon) | Significance (α=0.05) |
|---|---|---|---|---|---|---|---|---|
| **Hypervolume** | 0.915906 | 0.915922 | 0.022969 | 0.154050 | 0.155256 | 0.047781 | 1.8626e-09 | ✅ Significant |
| **Spacing** | 36.303402 | 31.598759 | 16.680287 | 22.392637 | 20.098970 | 9.125499 | 2.8326e-04 | ✅ Significant |
| **STE** | 0.006879 | 0.006041 | 0.003208 | 0.020333 | 0.019381 | 0.006977 | 1.8626e-09 | ✅ Significant |
| **GD** | 0.041120 | 0.041088 | 0.013932 | 0.098328 | 0.094865 | 0.019850 | 1.8626e-09 | ✅ Significant |
| **IGD** | 0.040684 | 0.039680 | 0.011654 | 0.574980 | 0.573633 | 0.042115 | 1.8626e-09 | ✅ Significant |
| **Coverage** | 0.982609 | 1.000000 | 0.093655 | 0.000000 | 0.000000 | 0.000000 | 6.7988e-08 | ✅ Significant |

## Execution Times Analysis and Statistical Tests

The following details the total execution times (in seconds), the obtained Speedup and the statistical comparisons using the Wilcoxon signed-rank test.

### Total Times Summary (seconds)

| Component | Mean (s) | Median (s) | Std. Dev. (s) |
|---|---|---|---|
| **Decomposition (Parallel Total)** | 100.8477 s | 100.8003 s | 0.4742 s |
| &nbsp;&nbsp;&nbsp;&nbsp;*└─ Slowest NSGA-II* | 93.5241 s | 93.4841 s | 0.4413 s |
| &nbsp;&nbsp;&nbsp;&nbsp;*└─ Slowest Aggregation* | 7.3236 s | 7.3048 s | 0.0978 s |
| **Traditional (Sequential Total)** | 278.1379 s | 278.2609 s | 2.1633 s |

### Speedup and Wilcoxon Comparison (Decomp vs Traditional)

Two statistical comparisons for execution time are presented:

#### 1. Total Times Comparison (Parallel Total with Aggregation vs Traditional Sequential)
Compares the full parallel time of decomposition (slowest NSGA-II + slowest aggregation step) against the total traditional sequential time:

| Metric | Value |
|---|---|
| **p-value (Wilcoxon)** | 1.8626e-09 |
| **Significance (α=0.05)** | ✅ Significant |
| **Mean Speedup** | 2.76x |
| **Median Speedup** | 2.76x |
| **Speedup Standard Deviation** | 0.02 |

#### 2. Genetic Algorithm Comparison (Only Slowest NSGA-II from Decomp vs Traditional)
Compares only the times dedicated to NSGA-II evolution, isolating the impact of the hierarchical aggregation operator:

| Metric | Value |
|---|---|
| **p-value (Wilcoxon)** | 1.8626e-09 |
| **Significance (α=0.05)** | ✅ Significant |
| **Mean Speedup** | 2.97x |
| **Median Speedup** | 2.98x |
| **Speedup Standard Deviation** | 0.03 |

