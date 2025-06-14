# Quantum Optimizer Benchmark Report

Generated on: 2025-03-12_12-09-57

## Overall Performance Ranking

| Optimizer                            | Function        |   Dimension |   Mean Value |   Mean Time (s) |   Mean Iterations | Success Rate   |   Mean Param Error |   Mean Value Error |
|:-------------------------------------|:----------------|------------:|-------------:|----------------:|------------------:|:---------------|-------------------:|-------------------:|
| Noise-Aware BFGS                     | Sphere Function |           5 |  1.30005e-11 |            0    |                 2 | 100.0%         |        3.60267e-06 |        1.30005e-11 |
| Gradient-Free nelder-mead            | Sphere Function |           5 |  4.17022e-09 |            0    |               445 | 100.0%         |        6.40201e-05 |        4.17022e-09 |
| Gradient-Free differential_evolution | Sphere Function |           5 |  0           |            0.25 |             15606 | 100.0%         |        0           |        0           |
| Adaptive Optimizer                   | Sphere Function |           5 |  9.97303e-17 |            0.01 |               502 | 100.0%         |        9.79195e-09 |        9.97303e-17 |
| Parallel Tempering                   | Sphere Function |           5 |  6.64945e-13 |            0.03 |              2472 | 0.0%           |        7.58211e-07 |        6.64945e-13 |

## Detailed Results by Function

### Sphere Function (5d)

Description: Simple convex quadratic function

| Optimizer                            |   Min Value |   Mean Value |   Std Value |   Mean Time (s) |   Mean Iterations | Success Rate   |   Mean Param Error |
|:-------------------------------------|------------:|-------------:|------------:|----------------:|------------------:|:---------------|-------------------:|
| Noise-Aware BFGS                     | 1.17875e-11 |  1.30005e-11 | 1.05346e-12 |            0    |                 2 | 100.0%         |        3.60267e-06 |
| Gradient-Free nelder-mead            | 2.75033e-09 |  4.17022e-09 | 1.04807e-09 |            0    |               445 | 100.0%         |        6.40201e-05 |
| Gradient-Free differential_evolution | 0           |  0           | 0           |            0.25 |             15606 | 100.0%         |        0           |
| Adaptive Optimizer                   | 4.92491e-17 |  9.97303e-17 | 3.56956e-17 |            0.01 |               502 | 100.0%         |        9.79195e-09 |
| Parallel Tempering                   | 1.52645e-13 |  6.64945e-13 | 4.59547e-13 |            0.03 |              2472 | 0.0%           |        7.58211e-07 |

