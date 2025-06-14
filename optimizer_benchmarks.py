#!/usr/bin/env python3
"""
Benchmark Suite for Quantum Optimizers

This module provides tools for benchmarking and comparing different optimization
strategies on standard test functions and quantum tasks.
"""

import json
import os
import pickle
import time
from datetime import datetime
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

# Import the optimizers
from advanced_optimizers import QuantumOptimizer, create_optimizer


class BenchmarkSuite:
    """
    A suite for benchmarking different optimizers on standard test functions.

    This class provides tools to:
    1. Run various optimizers on standard benchmark functions
    2. Generate statistics and visualizations of optimizer performance
    3. Save and load benchmark results for comparison
    4. Generate comprehensive reports on optimizer performance
    """

    def __init__(self, name: str = "Quantum Optimizer Benchmark"):
        """
        Initialize the benchmark suite.

        Args:
            name: Name of the benchmark suite
        """
        self.name = name
        self.results = {}
        self.test_functions = self._register_test_functions()
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = f"benchmark_results_{self.timestamp}"

    def _register_test_functions(self) -> Dict[str, Dict[str, Any]]:
        """
        Register standard test functions with the benchmark suite.

        Returns:
            Dictionary of test functions with metadata
        """
        test_functions = {}

        # Rosenbrock function (classic banana-shaped function)
        test_functions["rosenbrock"] = {
            "func": lambda x: sum(
                100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
                for i in range(len(x) - 1)
            ),
            "dim_range": (2, 20),  # Min and max dimensions
            "bounds": lambda dim: [(-5, 10) for _ in range(dim)],
            "global_minimum": lambda dim: np.ones(dim),
            "global_value": 0.0,
            "name": "Rosenbrock Function",
            "description": "Classic non-convex optimization test function shaped like a narrow, curved valley",
            "difficulty": "Medium",
        }

        # Sphere function (simple convex quadratic)
        test_functions["sphere"] = {
            "func": lambda x: sum(xi**2 for xi in x),
            "dim_range": (1, 100),
            "bounds": lambda dim: [(-10, 10) for _ in range(dim)],
            "global_minimum": lambda dim: np.zeros(dim),
            "global_value": 0.0,
            "name": "Sphere Function",
            "description": "Simple convex quadratic function",
            "difficulty": "Easy",
        }

        # Rastrigin function (highly multimodal)
        test_functions["rastrigin"] = {
            "func": lambda x: 10 * len(x)
            + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x),
            "dim_range": (1, 50),
            "bounds": lambda dim: [(-5.12, 5.12) for _ in range(dim)],
            "global_minimum": lambda dim: np.zeros(dim),
            "global_value": 0.0,
            "name": "Rastrigin Function",
            "description": "Highly multimodal function with many local minima",
            "difficulty": "Hard",
        }

        # Ackley function (many local minima)
        test_functions["ackley"] = {
            "func": lambda x: -20
            * np.exp(-0.2 * np.sqrt(sum(xi**2 for xi in x) / len(x)))
            - np.exp(sum(np.cos(2 * np.pi * xi) for xi in x) / len(x))
            + 20
            + np.e,
            "dim_range": (1, 50),
            "bounds": lambda dim: [(-32.768, 32.768) for _ in range(dim)],
            "global_minimum": lambda dim: np.zeros(dim),
            "global_value": 0.0,
            "name": "Ackley Function",
            "description": "Multimodal function with an exponential term and cosine modulation",
            "difficulty": "Hard",
        }

        # Beale function (2D, several local minima)
        test_functions["beale"] = {
            "func": lambda x: (1.5 - x[0] + x[0] * x[1]) ** 2
            + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
            + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2,
            "dim_range": (2, 2),  # Fixed 2D
            "bounds": lambda dim: [(-4.5, 4.5), (-4.5, 4.5)],
            "global_minimum": lambda dim: np.array([3.0, 0.5]),
            "global_value": 0.0,
            "name": "Beale Function",
            "description": "2D function with sharp peaks",
            "difficulty": "Medium",
        }

        # Levy function (multimodal with many local minima)
        test_functions["levy"] = {
            "func": lambda x: np.sin(3 * np.pi * x[0]) ** 2
            + sum(
                (x[i - 1] - 1) ** 2 * (1 + 10 * np.sin(3 * np.pi * x[i]) ** 2)
                for i in range(1, len(x))
            )
            + (x[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[-1]) ** 2),
            "dim_range": (2, 20),
            "bounds": lambda dim: [(-10, 10) for _ in range(dim)],
            "global_minimum": lambda dim: np.ones(dim),
            "global_value": 0.0,
            "name": "Levy Function",
            "description": "Multimodal function with many local minima",
            "difficulty": "Hard",
        }

        # Booth function (simple 2D function)
        test_functions["booth"] = {
            "func": lambda x: (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2,
            "dim_range": (2, 2),  # Fixed 2D
            "bounds": lambda dim: [(-10, 10), (-10, 10)],
            "global_minimum": lambda dim: np.array([1.0, 3.0]),
            "global_value": 0.0,
            "name": "Booth Function",
            "description": "Simple 2D quadratic function",
            "difficulty": "Easy",
        }

        # Noisy quadratic (to test resilience to noise)
        test_functions["noisy_quadratic"] = {
            "func": lambda x: sum((xi - 1) ** 2 for xi in x) + np.random.normal(0, 0.1),
            "dim_range": (1, 50),
            "bounds": lambda dim: [(-10, 10) for _ in range(dim)],
            "global_minimum": lambda dim: np.ones(dim),
            "global_value": 0.0,  # Approximate due to noise
            "name": "Noisy Quadratic Function",
            "description": "Quadratic function with Gaussian noise to test robustness",
            "difficulty": "Medium",
        }

        return test_functions

    def list_test_functions(self) -> pd.DataFrame:
        """
        List all available test functions with descriptions.

        Returns:
            DataFrame with test function information
        """
        data = []
        for func_id, func_info in self.test_functions.items():
            dim_range = f"{func_info['dim_range'][0]}-{func_info['dim_range'][1]}"
            if func_info["dim_range"][0] == func_info["dim_range"][1]:
                dim_range = str(func_info["dim_range"][0])

            data.append(
                {
                    "ID": func_id,
                    "Name": func_info["name"],
                    "Dimensions": dim_range,
                    "Difficulty": func_info["difficulty"],
                    "Description": func_info["description"],
                }
            )

        return pd.DataFrame(data)

    def run_benchmark(
        self,
        optimizers: List[QuantumOptimizer],
        function_ids: List[str] = None,
        dimensions: List[int] = None,
        runs_per_config: int = 3,
        max_iterations: int = 100,
        save_results: bool = True,
    ) -> Dict:
        """
        Run benchmarks on the specified optimizers and functions.

        Args:
            optimizers: List of optimizer instances to benchmark
            function_ids: List of function IDs to benchmark (default: all)
            dimensions: List of dimensions to test (default: min dim for each function)
            runs_per_config: Number of runs per configuration with different starting points
            max_iterations: Maximum iterations per optimization run
            save_results: Whether to save results to files

        Returns:
            Dictionary of benchmark results
        """
        # Create output directory if needed
        if save_results and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Default to all test functions if none specified
        if function_ids is None:
            function_ids = list(self.test_functions.keys())

        # Initialize results structure
        benchmark_results = {
            "name": self.name,
            "timestamp": self.timestamp,
            "configurations": [],
            "summary": {},
            "raw_data": {},
        }

        # Track benchmark progress
        total_configs = (
            len(function_ids)
            * len(optimizers)
            * (len(dimensions) if dimensions else 1)
            * runs_per_config
        )
        completed = 0

        print(f"Starting benchmark with {total_configs} total optimization runs...")
        start_time = time.time()

        # Run benchmarks for each combination
        for func_id in function_ids:
            func_info = self.test_functions[func_id]
            func = func_info["func"]

            # Create a wrapper function to handle 'shots' parameter from NoiseAwareOptimizer
            def wrapped_func(params, **kwargs):
                # Ignore additional kwargs like 'shots' that might be passed
                return func(params)

            # Determine dimensions to test
            if dimensions is None:
                # Default to minimum dimension for this function
                test_dimensions = [func_info["dim_range"][0]]
            else:
                # Filter dimensions to be within allowed range for this function
                min_dim, max_dim = func_info["dim_range"]
                test_dimensions = [d for d in dimensions if min_dim <= d <= max_dim]
                if not test_dimensions:
                    print(f"Warning: No valid dimensions for {func_id}, skipping...")
                    continue

            for dim in test_dimensions:
                # Get bounds and expected optimum
                bounds = func_info["bounds"](dim)
                global_minimum = func_info["global_minimum"](dim)
                global_value = func_info["global_value"]

                # Create config ID
                config_id = f"{func_id}_{dim}d"
                benchmark_results["raw_data"][config_id] = {}

                for optimizer in optimizers:
                    opt_name = optimizer.name
                    benchmark_results["raw_data"][config_id][opt_name] = []

                    for run in range(runs_per_config):
                        # Generate consistent but different initial points for each run
                        np.random.seed(42 + run)
                        if all(isinstance(b, tuple) and len(b) == 2 for b in bounds):
                            # Handle list of tuples bounds format
                            initial_params = np.array(
                                [np.random.uniform(low, high) for low, high in bounds]
                            )
                        else:
                            # Fallback to basic bounds
                            initial_params = np.random.uniform(-1, 1, dim)

                        # Reset optimizer for a clean run
                        optimizer.reset()

                        print(
                            f"Running: {func_info['name']} ({dim}D) - {opt_name} - Run {run + 1}/{runs_per_config}"
                        )

                        # Run optimization
                        try:
                            result = optimizer.minimize(
                                wrapped_func,
                                initial_params,
                                max_iterations=max_iterations,
                            )

                            # Calculate accuracy (distance from known global minimum)
                            if global_minimum is not None:
                                param_error = np.linalg.norm(
                                    result["params"] - global_minimum
                                )
                                value_error = abs(result["value"] - global_value)
                            else:
                                param_error = None
                                value_error = None

                            # Record detailed results
                            run_result = {
                                "run_id": run,
                                "initial_params": initial_params.tolist(),
                                "final_params": result["params"].tolist()
                                if isinstance(result["params"], np.ndarray)
                                else result["params"],
                                "iterations": result["iterations"],
                                "function_evaluations": len(
                                    result["history"]["values"]
                                ),
                                "value": result["value"],
                                "time": result["time"],
                                "converged": result["converged"],
                                "param_error": param_error,
                                "value_error": value_error,
                            }

                            benchmark_results["raw_data"][config_id][opt_name].append(
                                run_result
                            )

                        except Exception as e:
                            print(f"Error during optimization: {str(e)}")
                            # Record failure
                            run_result = {
                                "run_id": run,
                                "error": str(e),
                                "failed": True,
                            }
                            benchmark_results["raw_data"][config_id][opt_name].append(
                                run_result
                            )

                        # Update progress
                        completed += 1
                        progress = completed / total_configs * 100
                        elapsed = time.time() - start_time
                        est_total = elapsed / (completed / total_configs)
                        est_remaining = est_total - elapsed

                        print(
                            f"Progress: {progress:.1f}% - Est. remaining: {est_remaining / 60:.1f} min"
                        )

        # Generate summary statistics
        print("Generating summary statistics...")
        self._generate_summary(benchmark_results)

        # Save results if requested
        if save_results:
            results_file = os.path.join(self.output_dir, "benchmark_results.pkl")
            with open(results_file, "wb") as f:
                pickle.dump(benchmark_results, f)

            # Also save a readable JSON version
            json_file = os.path.join(self.output_dir, "benchmark_results.json")
            with open(json_file, "w") as f:
                # Convert numpy arrays to lists for JSON serialization
                json_results = self._prepare_for_json(benchmark_results)
                json.dump(json_results, f, indent=2)

            print(f"Results saved to {self.output_dir}/")

        # Store results in the instance
        self.results = benchmark_results

        return benchmark_results

    def _prepare_for_json(self, obj):
        """Helper method to prepare results for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(v) for v in obj]
        elif isinstance(obj, tuple):
            return [self._prepare_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj

    def _generate_summary(self, results: Dict):
        """Generate summary statistics for the benchmark results."""
        summary = {}

        # Process each configuration
        for config_id, config_data in results["raw_data"].items():
            # Split config_id safely
            parts = config_id.split("_")
            if len(parts) == 2:
                func_id, dim_str = parts
            else:
                func_id = "_".join(parts[:-1])
                dim_str = parts[-1]

            # Extract just the numeric part from the dimension string
            dim = int(dim_str.replace("d", ""))

            summary[config_id] = {}

            # For each optimizer
            for opt_name, runs in config_data.items():
                # Filter out failed runs
                valid_runs = [r for r in runs if "failed" not in r or not r["failed"]]

                if not valid_runs:
                    summary[config_id][opt_name] = {"status": "all_failed"}
                    continue

                # Calculate statistics
                values = [r["value"] for r in valid_runs]
                times = [r["time"] for r in valid_runs]
                iterations = [r["iterations"] for r in valid_runs]
                successes = [r["converged"] for r in valid_runs if "converged" in r]

                # Error metrics if available
                if (
                    "param_error" in valid_runs[0]
                    and valid_runs[0]["param_error"] is not None
                ):
                    param_errors = [r["param_error"] for r in valid_runs]
                    value_errors = [r["value_error"] for r in valid_runs]
                else:
                    param_errors = None
                    value_errors = None

                # Store statistics
                opt_summary = {
                    "mean_value": np.mean(values),
                    "std_value": np.std(values),
                    "min_value": min(values),
                    "max_value": max(values),
                    "mean_time": np.mean(times),
                    "std_time": np.std(times),
                    "mean_iterations": np.mean(iterations),
                    "std_iterations": np.std(iterations),
                    "success_rate": sum(successes) / len(successes) if successes else 0,
                    "run_count": len(valid_runs),
                    "failure_count": len(runs) - len(valid_runs),
                }

                if param_errors:
                    opt_summary["mean_param_error"] = np.mean(param_errors)
                    opt_summary["mean_value_error"] = np.mean(value_errors)

                summary[config_id][opt_name] = opt_summary

        # Add to results
        results["summary"] = summary

    def generate_report(self, output_file: str = None) -> str:
        """
        Generate a comprehensive report of benchmark results.

        Args:
            output_file: File to save the report (default: in output_dir)

        Returns:
            Report as a string
        """
        if not self.results:
            return "No benchmark results available. Run benchmarks first."

        # Generate report
        report = f"# {self.name} Report\n\n"
        report += f"Generated on: {self.timestamp}\n\n"

        # Overall performance ranking
        report += "## Overall Performance Ranking\n\n"

        # Collect data for ranking table
        ranking_data = []
        for config_id, config_summary in self.results["summary"].items():
            # Split config_id safely
            parts = config_id.split("_")
            if len(parts) == 2:
                func_id, dim_str = parts
            else:
                func_id = "_".join(parts[:-1])
                dim_str = parts[-1]

            func_info = self.test_functions[func_id]

            for opt_name, stats in config_summary.items():
                if "status" in stats and stats["status"] == "all_failed":
                    continue

                ranking_data.append(
                    {
                        "Optimizer": opt_name,
                        "Function": func_info["name"],
                        "Dimension": dim_str.replace("d", ""),
                        "Mean Value": f"{stats['mean_value']:.6e}",
                        "Mean Time (s)": f"{stats['mean_time']:.2f}",
                        "Mean Iterations": f"{stats['mean_iterations']:.1f}",
                        "Success Rate": f"{stats['success_rate'] * 100:.1f}%",
                    }
                )

                if "mean_param_error" in stats:
                    ranking_data[-1]["Mean Param Error"] = (
                        f"{stats['mean_param_error']:.6e}"
                    )
                    ranking_data[-1]["Mean Value Error"] = (
                        f"{stats['mean_value_error']:.6e}"
                    )

        # Create DataFrame and sort by performance metrics
        df = pd.DataFrame(ranking_data)
        report += (
            tabulate(df, headers="keys", tablefmt="pipe", showindex=False) + "\n\n"
        )

        # Detailed results by function
        report += "## Detailed Results by Function\n\n"

        for config_id, config_summary in self.results["summary"].items():
            # Split config_id safely
            parts = config_id.split("_")
            if len(parts) == 2:
                func_id, dim_str = parts
            else:
                func_id = "_".join(parts[:-1])
                dim_str = parts[-1]

            func_info = self.test_functions[func_id]

            report += f"### {func_info['name']} ({dim_str})\n\n"
            report += f"Description: {func_info['description']}\n\n"

            # Collect data for this function's table
            func_data = []
            for opt_name, stats in config_summary.items():
                if "status" in stats and stats["status"] == "all_failed":
                    func_data.append(
                        {"Optimizer": opt_name, "Status": "All runs failed"}
                    )
                    continue

                row = {
                    "Optimizer": opt_name,
                    "Min Value": f"{stats['min_value']:.6e}",
                    "Mean Value": f"{stats['mean_value']:.6e}",
                    "Std Value": f"{stats['std_value']:.6e}",
                    "Mean Time (s)": f"{stats['mean_time']:.2f}",
                    "Mean Iterations": f"{stats['mean_iterations']:.1f}",
                    "Success Rate": f"{stats['success_rate'] * 100:.1f}%",
                }

                if "mean_param_error" in stats:
                    row["Mean Param Error"] = f"{stats['mean_param_error']:.6e}"

                func_data.append(row)

            df = pd.DataFrame(func_data)
            report += (
                tabulate(df, headers="keys", tablefmt="pipe", showindex=False) + "\n\n"
            )

        # Save report if output_file is provided
        if output_file is None and self.output_dir:
            output_file = os.path.join(self.output_dir, "benchmark_report.md")

        if output_file:
            with open(output_file, "w") as f:
                f.write(report)
            print(f"Report saved to {output_file}")

        return report

    def plot_results(
        self,
        plot_type: str = "convergence",
        config_ids: List[str] = None,
        optimizer_names: List[str] = None,
        save_plots: bool = True,
    ) -> None:
        """
        Generate plots to visualize benchmark results.

        Args:
            plot_type: Type of plot ('convergence', 'performance', 'accuracy', 'all')
            config_ids: List of configuration IDs to plot (default: all)
            optimizer_names: List of optimizer names to include (default: all)
            save_plots: Whether to save plots to files
        """
        if not self.results:
            print("No benchmark results available. Run benchmarks first.")
            return

        # Create plots directory if needed
        plots_dir = os.path.join(self.output_dir, "plots")
        if save_plots and not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        # Default to all configurations
        if config_ids is None:
            config_ids = list(self.results["raw_data"].keys())

        # Set of plot types to generate
        plot_types = (
            ["convergence", "performance", "accuracy"]
            if plot_type == "all"
            else [plot_type]
        )

        for config_id in config_ids:
            if config_id not in self.results["raw_data"]:
                print(f"Warning: Configuration {config_id} not found in results.")
                continue

            # Split config_id safely
            parts = config_id.split("_")
            if len(parts) == 2:
                func_id, dim_str = parts
            else:
                func_id = "_".join(parts[:-1])
                dim_str = parts[-1]

            func_info = self.test_functions[func_id]

            config_data = self.results["raw_data"][config_id]

            # Filter optimizers if specified
            if optimizer_names:
                config_data = {
                    k: v for k, v in config_data.items() if k in optimizer_names
                }

            # Generate requested plot types
            for ptype in plot_types:
                if ptype == "convergence":
                    self._plot_convergence(
                        config_id,
                        func_info,
                        config_data,
                        save_dir=plots_dir if save_plots else None,
                    )
                elif ptype == "performance":
                    self._plot_performance(
                        config_id,
                        func_info,
                        config_data,
                        save_dir=plots_dir if save_plots else None,
                    )
                elif ptype == "accuracy":
                    self._plot_accuracy(
                        config_id,
                        func_info,
                        config_data,
                        save_dir=plots_dir if save_plots else None,
                    )

    def _plot_convergence(
        self, config_id: str, func_info: Dict, config_data: Dict, save_dir: str = None
    ):
        """Generate convergence plots for a specific configuration."""
        plt.figure(figsize=(12, 6))

        for opt_name, runs in config_data.items():
            # Filter out failed runs
            valid_runs = [r for r in runs if "failed" not in r or not r["failed"]]
            if not valid_runs:
                continue

            # Use first run for plotting (could average, but histories might have different lengths)
            first_run = valid_runs[0]

            if "history" in first_run and "values" in first_run["history"]:
                values = first_run["history"]["values"]
                plt.plot(values, label=f"{opt_name}")
            else:
                # Fallback if history isn't available
                plt.plot(
                    [first_run["value"]], marker="o", label=f"{opt_name} (final only)"
                )

        # Extract dimension safely from config_id
        parts = config_id.split("_")
        if len(parts) >= 2:
            dim_str = parts[-1]
        else:
            dim_str = "unknown_dim"

        plt.title(f"Convergence: {func_info['name']} ({dim_str})")
        plt.xlabel("Iterations")
        plt.ylabel("Function Value")
        plt.yscale("log" if plt.ylim()[0] > 0 else "symlog")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend()

        if save_dir:
            plt.savefig(
                os.path.join(save_dir, f"convergence_{config_id}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
        else:
            plt.show()

    def _plot_performance(
        self, config_id: str, func_info: Dict, config_data: Dict, save_dir: str = None
    ):
        """Generate performance comparison plots for a specific configuration."""
        # Extract data
        data = []
        for opt_name, runs in config_data.items():
            # Filter out failed runs
            valid_runs = [r for r in runs if "failed" not in r or not r["failed"]]
            if not valid_runs:
                continue

            # Calculate statistics
            times = [r["time"] for r in valid_runs]
            iterations = [r["iterations"] for r in valid_runs]
            values = [r["value"] for r in valid_runs]

            data.append(
                {
                    "Optimizer": opt_name,
                    "Mean Time": np.mean(times),
                    "Mean Iterations": np.mean(iterations),
                    "Mean Value": np.mean(values),
                }
            )

        df = pd.DataFrame(data)
        if df.empty:
            return

        # Create three subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot time comparison
        axes[0].bar(df["Optimizer"], df["Mean Time"], color="skyblue")
        axes[0].set_title("Mean Optimization Time")
        axes[0].set_ylabel("Time (seconds)")
        axes[0].set_xticklabels(df["Optimizer"], rotation=45, ha="right")
        axes[0].grid(axis="y", linestyle="--", alpha=0.7)

        # Plot iterations comparison
        axes[1].bar(df["Optimizer"], df["Mean Iterations"], color="lightgreen")
        axes[1].set_title("Mean Iterations")
        axes[1].set_ylabel("Iterations")
        axes[1].set_xticklabels(df["Optimizer"], rotation=45, ha="right")
        axes[1].grid(axis="y", linestyle="--", alpha=0.7)

        # Plot final value comparison
        axes[2].bar(df["Optimizer"], df["Mean Value"], color="salmon")
        axes[2].set_title("Mean Final Value")
        axes[2].set_ylabel("Function Value")
        axes[2].set_xticklabels(df["Optimizer"], rotation=45, ha="right")
        axes[2].grid(axis="y", linestyle="--", alpha=0.7)

        # Extract dimension safely from config_id
        parts = config_id.split("_")
        if len(parts) >= 2:
            dim_str = parts[-1]
        else:
            dim_str = "unknown_dim"

        plt.suptitle(f"Performance Comparison: {func_info['name']} ({dim_str})")
        plt.tight_layout()

        if save_dir:
            plt.savefig(
                os.path.join(save_dir, f"performance_{config_id}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
        else:
            plt.show()

    def _plot_accuracy(
        self, config_id: str, func_info: Dict, config_data: Dict, save_dir: str = None
    ):
        """Generate accuracy plots for a specific configuration."""
        # Check if we have accuracy data
        has_accuracy = False
        for runs in config_data.values():
            valid_runs = [r for r in runs if "failed" not in r or not r["failed"]]
            if (
                valid_runs
                and "param_error" in valid_runs[0]
                and valid_runs[0]["param_error"] is not None
            ):
                has_accuracy = True
                break

        if not has_accuracy:
            return

        # Extract accuracy data
        data = []
        for opt_name, runs in config_data.items():
            # Filter out failed runs
            valid_runs = [r for r in runs if "failed" not in r or not r["failed"]]
            if not valid_runs or "param_error" not in valid_runs[0]:
                continue

            # Calculate statistics
            param_errors = [r["param_error"] for r in valid_runs]
            value_errors = [r["value_error"] for r in valid_runs]

            data.append(
                {
                    "Optimizer": opt_name,
                    "Mean Parameter Error": np.mean(param_errors),
                    "Mean Value Error": np.mean(value_errors),
                }
            )

        df = pd.DataFrame(data)
        if df.empty:
            return

        # Create two subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot parameter error
        axes[0].bar(df["Optimizer"], df["Mean Parameter Error"], color="purple")
        axes[0].set_title("Mean Parameter Error")
        axes[0].set_ylabel("Error (L2 distance)")
        axes[0].set_xticklabels(df["Optimizer"], rotation=45, ha="right")
        axes[0].set_yscale("log")
        axes[0].grid(axis="y", linestyle="--", alpha=0.7)

        # Plot value error
        axes[1].bar(df["Optimizer"], df["Mean Value Error"], color="orange")
        axes[1].set_title("Mean Value Error")
        axes[1].set_ylabel("Error (|f(x) - f(x*)|)")
        axes[1].set_xticklabels(df["Optimizer"], rotation=45, ha="right")
        axes[1].set_yscale("log")
        axes[1].grid(axis="y", linestyle="--", alpha=0.7)

        # Extract dimension safely from config_id
        parts = config_id.split("_")
        if len(parts) >= 2:
            dim_str = parts[-1]
        else:
            dim_str = "unknown_dim"

        plt.suptitle(f"Accuracy Comparison: {func_info['name']} ({dim_str})")
        plt.tight_layout()

        if save_dir:
            plt.savefig(
                os.path.join(save_dir, f"accuracy_{config_id}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
        else:
            plt.show()

    def load_results(self, filepath: str):
        """
        Load benchmark results from a file.

        Args:
            filepath: Path to the results file
        """
        with open(filepath, "rb") as f:
            self.results = pickle.load(f)

        # Extract timestamp from results
        self.timestamp = self.results.get("timestamp", "unknown_time")
        self.output_dir = f"benchmark_results_{self.timestamp}"

        print(f"Loaded benchmark results from {filepath}")

        return self.results


# Example usage function
def run_example_benchmark():
    """Run a simple example benchmark to demonstrate usage."""
    print("Running example benchmark...")

    # Create benchmark suite
    benchmark = BenchmarkSuite("Example Quantum Optimizer Benchmark")

    # Create optimizers to test
    optimizers = [
        create_optimizer("bfgs"),
        create_optimizer("noise_aware"),
        create_optimizer("gradient_free", method="nelder-mead"),
        create_optimizer("gradient_free", method="differential_evolution"),
        create_optimizer("adaptive"),
        create_optimizer("parallel_tempering"),
    ]

    # List available test functions
    print("\nAvailable test functions:")
    print(benchmark.list_test_functions())

    # Run benchmarks on a subset of functions with 2D and 5D variants
    results = benchmark.run_benchmark(
        optimizers=optimizers,
        function_ids=["rosenbrock", "sphere", "rastrigin"],
        dimensions=[2, 5],
        runs_per_config=2,  # Reduced for example
        max_iterations=50,  # Reduced for example
    )

    # Generate report
    report = benchmark.generate_report()
    print("\nBenchmark Report:")
    print(report[:500] + "...\n[Report truncated]")

    # Generate plots
    benchmark.plot_results(plot_type="all")

    print(
        "\nExample benchmark completed! Check the benchmark_results_* directory for full results."
    )


if __name__ == "__main__":
    run_example_benchmark()
