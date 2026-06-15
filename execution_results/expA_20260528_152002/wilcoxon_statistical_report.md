# Global Statistical Analysis Report (30 Runs)

This report statistically analyzes the performance of Decomposition mode vs Traditional mode.

## Performance Metrics (Mean, Median, Standard Deviation and Wilcoxon)

| Metric | Mean Decomp. | Median Decomp. | Std. Dev. Decomp. | Mean Trad. | Median Trad. | Std. Dev. Trad. | p-value (Wilcoxon) | Significance (α=0.05) |
|---|---|---|---|---|---|---|---|---|
| **Hypervolume** | 1.071086 | 1.071863 | 0.007100 | 0.949550 | 0.954742 | 0.030603 | 1.8626e-09 | ✅ Significant |
| **Spacing** | 20.320955 | 17.294748 | 10.141681 | 9.738182 | 8.174280 | 4.074725 | 3.1479e-07 | ✅ Significant |
| **STE** | 0.007214 | 0.006194 | 0.003621 | 0.005917 | 0.005349 | 0.002099 | 1.0943e-01 | ❌ Not Signif. |
| **GD** | 0.012867 | 0.012404 | 0.003377 | 0.003170 | 0.002825 | 0.001324 | 1.8626e-09 | ✅ Significant |
| **IGD** | 0.015825 | 0.015451 | 0.002940 | 0.078786 | 0.077332 | 0.014870 | 1.8626e-09 | ✅ Significant |
| **Coverage** | 0.057000 | 0.030000 | 0.062724 | 0.352871 | 0.360000 | 0.071355 | 1.7257e-06 | ✅ Significant |

## Execution Times Analysis and Statistical Tests

The following details the total execution times (in seconds), the obtained Speedup and the statistical comparisons using the Wilcoxon signed-rank test.

### Total Times Summary (seconds)

| Component | Mean (s) | Median (s) | Std. Dev. (s) |
|---|---|---|---|
| **Decomposition (Parallel Total)** | 24.7845 s | 24.7831 s | 0.1395 s |
| &nbsp;&nbsp;&nbsp;&nbsp;*└─ Slowest NSGA-II* | 22.3163 s | 22.3117 s | 0.1229 s |
| &nbsp;&nbsp;&nbsp;&nbsp;*└─ Slowest Aggregation* | 2.4682 s | 2.4765 s | 0.0529 s |
| **Traditional (Sequential Total)** | 49.0320 s | 49.0148 s | 0.4024 s |

### Speedup and Wilcoxon Comparison (Decomp vs Traditional)

Two statistical comparisons for execution time are presented:

#### 1. Total Times Comparison (Parallel Total with Aggregation vs Traditional Sequential)
Compares the full parallel time of decomposition (slowest NSGA-II + slowest aggregation step) against the total traditional sequential time:

| Metric | Value |
|---|---|
| **p-value (Wilcoxon)** | 1.8626e-09 |
| **Significance (α=0.05)** | ✅ Significant |
| **Mean Speedup** | 1.98x |
| **Median Speedup** | 1.98x |
| **Speedup Standard Deviation** | 0.02 |

#### 2. Genetic Algorithm Comparison (Only Slowest NSGA-II from Decomp vs Traditional)
Compares only the times dedicated to NSGA-II evolution, isolating the impact of the hierarchical aggregation operator:

| Metric | Value |
|---|---|
| **p-value (Wilcoxon)** | 1.8626e-09 |
| **Significance (α=0.05)** | ✅ Significant |
| **Mean Speedup** | 2.20x |
| **Median Speedup** | 2.20x |
| **Speedup Standard Deviation** | 0.02 |

