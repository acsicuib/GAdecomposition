# Global Statistical Analysis Report (30 Runs)

This report statistically analyzes the performance of Decomposition mode vs Traditional mode.

## Performance Metrics (Mean, Median, Standard Deviation and Wilcoxon)

| Metric | Mean Decomp. | Median Decomp. | Std. Dev. Decomp. | Mean Trad. | Median Trad. | Std. Dev. Trad. | p-value (Wilcoxon) | Significance (α=0.05) |
|---|---|---|---|---|---|---|---|---|
| **Hypervolume** | 0.909008 | 0.909413 | 0.001671 | 0.775285 | 0.778552 | 0.033640 | 1.8626e-09 | ✅ Significant |
| **Spacing** | 3.342719 | 3.178083 | 0.500337 | 1.565242 | 1.439897 | 0.406030 | 1.8626e-09 | ✅ Significant |
| **STE** | 0.008592 | 0.008169 | 0.001286 | 0.005670 | 0.005154 | 0.001606 | 1.3039e-07 | ✅ Significant |
| **GD** | 0.006291 | 0.006388 | 0.001166 | 0.007742 | 0.006264 | 0.004373 | 2.7101e-01 | ❌ Not Signif. |
| **IGD** | 0.012591 | 0.012335 | 0.001550 | 0.114425 | 0.114693 | 0.023101 | 1.8626e-09 | ✅ Significant |
| **Coverage** | 0.250225 | 0.235000 | 0.099321 | 0.254450 | 0.248077 | 0.059605 | 9.1930e-01 | ❌ Not Signif. |

## Execution Times Analysis and Statistical Tests

The following details the total execution times (in seconds), the obtained Speedup and the statistical comparisons using the Wilcoxon signed-rank test.

### Total Times Summary (seconds)

| Component | Mean (s) | Median (s) | Std. Dev. (s) |
|---|---|---|---|
| **Decomposition (Parallel Total)** | 23.7124 s | 23.6669 s | 0.4664 s |
| &nbsp;&nbsp;&nbsp;&nbsp;*└─ Slowest NSGA-II* | 22.1895 s | 22.1457 s | 0.4653 s |
| &nbsp;&nbsp;&nbsp;&nbsp;*└─ Slowest Aggregation* | 1.5228 s | 1.5236 s | 0.0748 s |
| **Traditional (Sequential Total)** | 48.5110 s | 48.7459 s | 0.7962 s |

### Speedup and Wilcoxon Comparison (Decomp vs Traditional)

Two statistical comparisons for execution time are presented:

#### 1. Total Times Comparison (Parallel Total with Aggregation vs Traditional Sequential)
Compares the full parallel time of decomposition (slowest NSGA-II + slowest aggregation step) against the total traditional sequential time:

| Metric | Value |
|---|---|
| **p-value (Wilcoxon)** | 1.8626e-09 |
| **Significance (α=0.05)** | ✅ Significant |
| **Mean Speedup** | 2.05x |
| **Median Speedup** | 2.05x |
| **Speedup Standard Deviation** | 0.04 |

#### 2. Genetic Algorithm Comparison (Only Slowest NSGA-II from Decomp vs Traditional)
Compares only the times dedicated to NSGA-II evolution, isolating the impact of the hierarchical aggregation operator:

| Metric | Value |
|---|---|
| **p-value (Wilcoxon)** | 1.8626e-09 |
| **Significance (α=0.05)** | ✅ Significant |
| **Mean Speedup** | 2.19x |
| **Median Speedup** | 2.19x |
| **Speedup Standard Deviation** | 0.04 |

