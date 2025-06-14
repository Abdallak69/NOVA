# Example Quantum Optimizer Benchmark Report

Generated on: 2025-03-12_12-05-52

## Overall Performance Ranking

| Optimizer                            | Function            |   Dimension |     Mean Value |   Mean Time (s) |   Mean Iterations | Success Rate   |   Mean Param Error |   Mean Value Error |
|:-------------------------------------|:--------------------|------------:|---------------:|----------------:|------------------:|:---------------|-------------------:|-------------------:|
| Gradient-Free nelder-mead            | Rosenbrock Function |           2 |    1.22783     |            0    |              94.5 | 0.0%           |        0.850226    |        1.22783     |
| Gradient-Free differential_evolution | Rosenbrock Function |           2 |    9.22036e-11 |            0.03 |            1539   | 0.0%           |        2.14879e-05 |        9.22036e-11 |
| Adaptive Optimizer                   | Rosenbrock Function |           2 |    0.0293238   |            0    |             195   | 0.0%           |        0.227132    |        0.0293238   |
| Parallel Tempering                   | Rosenbrock Function |           2 |    1.37943e-06 |            0.02 |             832.5 | 0.0%           |        0.00166242  |        1.37943e-06 |
| Gradient-Free nelder-mead            | Rosenbrock Function |           5 | 3748.96        |            0    |              81.5 | 0.0%           |        5.15147     |     3748.96        |
| Gradient-Free differential_evolution | Rosenbrock Function |           5 |    2.07796e-11 |            0.07 |            3987   | 0.0%           |        8.60906e-06 |        2.07796e-11 |
| Adaptive Optimizer                   | Rosenbrock Function |           5 |    2.42696     |            0.01 |             363   | 0.0%           |        2.23698     |        2.42696     |
| Parallel Tempering                   | Rosenbrock Function |           5 |    0.00498225  |            0.03 |            2007   | 0.0%           |        0.111335    |        0.00498225  |
| Gradient-Free nelder-mead            | Sphere Function     |           2 |    1.45867e-09 |            0    |              94   | 50.0%          |        3.46213e-05 |        1.45867e-09 |
| Gradient-Free differential_evolution | Sphere Function     |           2 |    5.18738e-17 |            0.02 |            1533   | 0.0%           |        7.20235e-09 |        5.18738e-17 |
| Adaptive Optimizer                   | Sphere Function     |           2 |    3.58048e-17 |            0    |             107   | 100.0%         |        5.85987e-09 |        3.58048e-17 |
| Parallel Tempering                   | Sphere Function     |           2 |    4.69178e-14 |            0    |              78   | 100.0%         |        1.613e-07   |        4.69178e-14 |
| Gradient-Free nelder-mead            | Sphere Function     |           5 |   28.7482      |            0    |              88.5 | 0.0%           |        5.35918     |       28.7482      |
| Gradient-Free differential_evolution | Sphere Function     |           5 |    2.56119e-14 |            0.06 |            3843   | 0.0%           |        1.60037e-07 |        2.56119e-14 |
| Adaptive Optimizer                   | Sphere Function     |           5 |    1.3019e-16  |            0    |             276   | 100.0%         |        1.1408e-08  |        1.3019e-16  |
| Parallel Tempering                   | Sphere Function     |           5 |    9.21095e-13 |            0    |             192   | 0.0%           |        9.41967e-07 |        9.21095e-13 |
| Gradient-Free nelder-mead            | Rastrigin Function  |           2 |   21.3914      |            0    |              59   | 100.0%         |        4.58778     |       21.3914      |
| Gradient-Free differential_evolution | Rastrigin Function  |           2 |    0           |            0.02 |            1533   | 0.0%           |        3.05713e-09 |        0           |
| Adaptive Optimizer                   | Rastrigin Function  |           2 |   20.894       |            0    |              61   | 0.0%           |        4.53848     |       20.894       |
| Parallel Tempering                   | Rastrigin Function  |           2 |   20.894       |            0    |             168   | 100.0%         |        4.53848     |       20.894       |
| Gradient-Free nelder-mead            | Rastrigin Function  |           5 |   46.7332      |            0    |              80   | 0.0%           |        6.78705     |       46.7332      |
| Gradient-Free differential_evolution | Rastrigin Function  |           5 |    2.98488     |            0.07 |            3879   | 0.0%           |        1.72332     |        2.98488     |
| Adaptive Optimizer                   | Rastrigin Function  |           5 |   45.7678      |            0    |             168   | 0.0%           |        6.74805     |       45.7678      |
| Parallel Tempering                   | Rastrigin Function  |           5 |   41.2906      |            0.01 |             837   | 0.0%           |        6.40005     |       41.2906      |

## Detailed Results by Function

### Rosenbrock Function (2d)

Description: Classic non-convex optimization test function shaped like a narrow, curved valley

| Optimizer                            | Status          |     Min Value |    Mean Value |     Std Value |   Mean Time (s) |   Mean Iterations | Success Rate   |   Mean Param Error |
|:-------------------------------------|:----------------|--------------:|--------------:|--------------:|----------------:|------------------:|:---------------|-------------------:|
| Noise-Aware BFGS                     | All runs failed | nan           | nan           | nan           |          nan    |             nan   | nan            |      nan           |
| Gradient-Free nelder-mead            | nan             |   8.18356e-10 |   1.22783     |   1.22783     |            0    |              94.5 | 0.0%           |        0.850226    |
| Gradient-Free differential_evolution | nan             |   9.22036e-11 |   9.22036e-11 |   0           |            0.03 |            1539   | 0.0%           |        2.14879e-05 |
| Adaptive Optimizer                   | nan             |   1.98047e-11 |   0.0293238   |   0.0293238   |            0    |             195   | 0.0%           |        0.227132    |
| Parallel Tempering                   | nan             |   2.71771e-12 |   1.37943e-06 |   1.37943e-06 |            0.02 |             832.5 | 0.0%           |        0.00166242  |

### Rosenbrock Function (5d)

Description: Classic non-convex optimization test function shaped like a narrow, curved valley

| Optimizer                            | Status          |     Min Value |     Mean Value |     Std Value |   Mean Time (s) |   Mean Iterations | Success Rate   |   Mean Param Error |
|:-------------------------------------|:----------------|--------------:|---------------:|--------------:|----------------:|------------------:|:---------------|-------------------:|
| Noise-Aware BFGS                     | All runs failed | nan           |  nan           |  nan          |          nan    |             nan   | nan            |      nan           |
| Gradient-Free nelder-mead            | nan             | 554.615       | 3748.96        | 3194.34       |            0    |              81.5 | 0.0%           |        5.15147     |
| Gradient-Free differential_evolution | nan             |   2.07796e-11 |    2.07796e-11 |    0          |            0.07 |            3987   | 0.0%           |        8.60906e-06 |
| Adaptive Optimizer                   | nan             |   0.920907    |    2.42696     |    1.50605    |            0.01 |             363   | 0.0%           |        2.23698     |
| Parallel Tempering                   | nan             |   0.00298465  |    0.00498225  |    0.00199761 |            0.03 |            2007   | 0.0%           |        0.111335    |

### Sphere Function (2d)

Description: Simple convex quadratic function

| Optimizer                            | Status          |     Min Value |    Mean Value |     Std Value |   Mean Time (s) |   Mean Iterations | Success Rate   |   Mean Param Error |
|:-------------------------------------|:----------------|--------------:|--------------:|--------------:|----------------:|------------------:|:---------------|-------------------:|
| Noise-Aware BFGS                     | All runs failed | nan           | nan           | nan           |          nan    |               nan | nan            |      nan           |
| Gradient-Free nelder-mead            | nan             |   3.42086e-10 |   1.45867e-09 |   1.11659e-09 |            0    |                94 | 50.0%          |        3.46213e-05 |
| Gradient-Free differential_evolution | nan             |   5.18738e-17 |   5.18738e-17 |   0           |            0.02 |              1533 | 0.0%           |        7.20235e-09 |
| Adaptive Optimizer                   | nan             |   2.16113e-17 |   3.58048e-17 |   1.41935e-17 |            0    |               107 | 100.0%         |        5.85987e-09 |
| Parallel Tempering                   | nan             |   2.79947e-16 |   4.69178e-14 |   4.66378e-14 |            0    |                78 | 100.0%         |        1.613e-07   |

### Sphere Function (5d)

Description: Simple convex quadratic function

| Optimizer                            | Status          |     Min Value |    Mean Value |     Std Value |   Mean Time (s) |   Mean Iterations | Success Rate   |   Mean Param Error |
|:-------------------------------------|:----------------|--------------:|--------------:|--------------:|----------------:|------------------:|:---------------|-------------------:|
| Noise-Aware BFGS                     | All runs failed | nan           | nan           | nan           |          nan    |             nan   | nan            |      nan           |
| Gradient-Free nelder-mead            | nan             |  26.974       |  28.7482      |   1.7742      |            0    |              88.5 | 0.0%           |        5.35918     |
| Gradient-Free differential_evolution | nan             |   2.56119e-14 |   2.56119e-14 |   0           |            0.06 |            3843   | 0.0%           |        1.60037e-07 |
| Adaptive Optimizer                   | nan             |   1.25256e-16 |   1.3019e-16  |   4.9348e-18  |            0    |             276   | 100.0%         |        1.1408e-08  |
| Parallel Tempering                   | nan             |   5.74776e-13 |   9.21095e-13 |   3.46319e-13 |            0    |             192   | 0.0%           |        9.41967e-07 |

### Rastrigin Function (2d)

Description: Highly multimodal function with many local minima

| Optimizer                            | Status          |   Min Value |   Mean Value |   Std Value |   Mean Time (s) |   Mean Iterations | Success Rate   |   Mean Param Error |
|:-------------------------------------|:----------------|------------:|-------------:|------------:|----------------:|------------------:|:---------------|-------------------:|
| Noise-Aware BFGS                     | All runs failed |    nan      |     nan      |   nan       |          nan    |               nan | nan            |      nan           |
| Gradient-Free nelder-mead            | nan             |     16.9142 |      21.3914 |     4.47724 |            0    |                59 | 100.0%         |        4.58778     |
| Gradient-Free differential_evolution | nan             |      0      |       0      |     0       |            0.02 |              1533 | 0.0%           |        3.05713e-09 |
| Adaptive Optimizer                   | nan             |     16.9142 |      20.894  |     3.97976 |            0    |                61 | 0.0%           |        4.53848     |
| Parallel Tempering                   | nan             |     16.9142 |      20.894  |     3.97976 |            0    |               168 | 100.0%         |        4.53848     |

### Rastrigin Function (5d)

Description: Highly multimodal function with many local minima

| Optimizer                            | Status          |   Min Value |   Mean Value |     Std Value |   Mean Time (s) |   Mean Iterations | Success Rate   |   Mean Param Error |
|:-------------------------------------|:----------------|------------:|-------------:|--------------:|----------------:|------------------:|:---------------|-------------------:|
| Noise-Aware BFGS                     | All runs failed |   nan       |    nan       | nan           |          nan    |               nan | nan            |          nan       |
| Gradient-Free nelder-mead            | nan             |    45.9849  |     46.7332  |   0.748315    |            0    |                80 | 0.0%           |            6.78705 |
| Gradient-Free differential_evolution | nan             |     2.98488 |      2.98488 |   0           |            0.07 |              3879 | 0.0%           |            1.72332 |
| Adaptive Optimizer                   | nan             |    45.7678  |     45.7678  |   6.10497e-05 |            0    |               168 | 0.0%           |            6.74805 |
| Parallel Tempering                   | nan             |    36.8133  |     41.2906  |   4.4773      |            0.01 |               837 | 0.0%           |            6.40005 |

