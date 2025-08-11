#!/usr/bin/env python3
"""
Advanced Optimization Strategies for Quantum Neural Networks

This module provides various advanced optimization methods for training quantum
neural networks, including gradient-free, noise-aware, and hybrid approaches.
"""

import pickle
import time
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.optimize import basinhopping, differential_evolution, minimize


class QuantumOptimizer:
    """Base class for quantum circuit optimizers."""

    def __init__(self, name: str = "base"):
        """
        Initialize the quantum optimizer.

        Args:
            name: Name of the optimizer
        """
        self.name = name
        self.history = {
            "params": [],
            "values": [],
            "times": [],
            "gradients": [],
            "step_sizes": [],
        }
        self.best_params = None
        self.best_value = float("inf")
        self.iterations = 0
        self.start_time = None
        self.total_time = 0.0
        self.converged = False

    def reset(self):
        """Reset the optimizer state."""
        self.history = {
            "params": [],
            "values": [],
            "times": [],
            "gradients": [],
            "step_sizes": [],
        }
        self.best_params = None
        self.best_value = float("inf")
        self.iterations = 0
        self.start_time = None
        self.total_time = 0.0
        self.converged = False

    def minimize(
        self,
        objective_fn: Callable,
        initial_params: np.ndarray,
        callback: Optional[Callable] = None,
        **kwargs,
    ) -> Dict:
        """
        Minimize an objective function.

        Args:
            objective_fn: Function to minimize
            initial_params: Initial parameter values
            callback: Optional callback function
            **kwargs: Additional arguments

        Returns:
            Optimization results
        """
        raise NotImplementedError("Subclasses must implement minimize method")

    def save_history(self, filepath: str):
        """
        Save optimization history to a file.

        Args:
            filepath: Path to save the history
        """
        with open(filepath, "wb") as f:
            pickle.dump(self.history, f)

    def load_history(self, filepath: str):
        """
        Load optimization history from a file.

        Args:
            filepath: Path to load the history from
        """
        with open(filepath, "rb") as f:
            self.history = pickle.load(f)  # noqa: S301

        # Update best parameters and value
        if self.history["values"]:
            best_idx = np.argmin(self.history["values"])
            self.best_value = self.history["values"][best_idx]
            self.best_params = self.history["params"][best_idx]
            self.iterations = len(self.history["values"])

    def __str__(self) -> str:
        """String representation of the optimizer."""
        return f"{self.name} Optimizer"


class NoiseAwareOptimizer(QuantumOptimizer):
    """
    Optimizer that is aware of quantum noise and tries to mitigate its effects.

    This optimizer uses repeated measurements and statistical techniques to
    handle the stochastic nature of quantum measurements.
    """

    def __init__(
        self,
        base_optimizer: str = "BFGS",
        shots_schedule: Optional[List[int]] = None,
        noise_adaptation: bool = True,
    ):
        """
        Initialize the noise-aware optimizer.

        Args:
            base_optimizer: Name of the base optimization method
            shots_schedule: Schedule for increasing shot counts during optimization
            noise_adaptation: Whether to adapt to noise levels
        """
        super().__init__(name=f"Noise-Aware {base_optimizer}")
        self.base_optimizer = base_optimizer
        self.shots_schedule = shots_schedule or [100, 500, 1000, 5000]
        self.noise_adaptation = noise_adaptation
        self.current_shots = self.shots_schedule[0]
        self.shot_phase = 0
        self.noise_level_estimate = 0.0

    def _noise_adapted_objective(
        self, objective_fn: Callable, params: np.ndarray
    ) -> float:
        """
        Evaluate objective function with noise adaptation.

        Args:
            objective_fn: Original objective function
            params: Parameters to evaluate

        Returns:
            Noise-adapted function value
        """
        # Determine how many evaluations to perform based on noise level
        n_evals = 1
        if self.noise_adaptation and self.noise_level_estimate > 0.01:
            # More noise -> more evaluations
            n_evals = min(5, int(10 * self.noise_level_estimate))

        # Evaluate multiple times and take the average
        values = []
        for _ in range(n_evals):
            values.append(objective_fn(params, shots=self.current_shots))

        # Update noise level estimate
        if len(values) > 1:
            self.noise_level_estimate = 0.9 * self.noise_level_estimate + 0.1 * np.std(
                values
            )

        return np.mean(values)

    def _callback_wrapper(
        self, original_callback: Optional[Callable], params: np.ndarray
    ):
        """
        Wrapper for the callback function to record history and update shots.

        Args:
            original_callback: Original callback function
            params: Current parameters
        """
        # Update iteration count
        self.iterations += 1

        # Evaluate objective (elapsed time computed via history timestamps)

        # Check if we should increase shots
        if self.shot_phase < len(self.shots_schedule) - 1:
            # Increase shots every N iterations, where N depends on the dimension
            phase_length = 10 * len(params)
            if self.iterations % phase_length == 0:
                self.shot_phase += 1
                self.current_shots = self.shots_schedule[self.shot_phase]

        # Call original callback if provided
        if original_callback:
            original_callback(params)

    def minimize(
        self,
        objective_fn: Callable,
        initial_params: np.ndarray,
        callback: Optional[Callable] = None,
        **kwargs,
    ) -> Dict:
        """
        Minimize an objective function with noise awareness.

        Args:
            objective_fn: Function to minimize
            initial_params: Initial parameter values
            callback: Optional callback function
            **kwargs: Additional arguments

        Returns:
            Optimization results
        """
        self.reset()
        self.start_time = time.time()
        self.current_shots = self.shots_schedule[0]
        self.shot_phase = 0

        # Create wrapper for objective function
        def wrapped_objective(params):
            value = self._noise_adapted_objective(objective_fn, params)

            # Update history
            current_time = time.time() - self.start_time
            self.history["params"].append(params.copy())
            self.history["values"].append(value)
            self.history["times"].append(current_time)

            # Update best value and parameters
            if value < self.best_value:
                self.best_value = value
                self.best_params = params.copy()

            return value

        # Create wrapper for callback
        def wrapped_callback(params):  # noqa: D401
            return self._callback_wrapper(callback, params)

        # Run the optimization
        max_iterations = kwargs.pop("max_iterations", 100)
        result = minimize(
            wrapped_objective,
            initial_params,
            method=self.base_optimizer,
            callback=wrapped_callback,
            options={"maxiter": max_iterations},
            **kwargs,
        )

        self.total_time = time.time() - self.start_time
        self.converged = result.success

        return {
            "params": self.best_params,
            "value": self.best_value,
            "iterations": self.iterations,
            "time": self.total_time,
            "history": self.history,
            "converged": self.converged,
            "noise_level": self.noise_level_estimate,
        }


class GradientFreeOptimizer(QuantumOptimizer):
    """
    Gradient-free optimizer for quantum circuits.

    Uses techniques like evolutionary algorithms, particle swarm, or Nelder-Mead
    that don't require gradient information and are more robust to noise.
    """

    def __init__(
        self, method: str = "differential_evolution", population_size: int = 15
    ):
        """
        Initialize the gradient-free optimizer.

        Args:
            method: Optimization method ('differential_evolution', 'nelder-mead', 'particle_swarm')
            population_size: Size of the population for population-based methods
        """
        super().__init__(name=f"Gradient-Free {method}")
        self.method = method
        self.population_size = population_size

    def minimize(
        self,
        objective_fn: Callable,
        initial_params: np.ndarray,
        callback: Optional[Callable] = None,
        **kwargs,
    ) -> Dict:
        """
        Minimize an objective function without using gradients.

        Args:
            objective_fn: Function to minimize
            initial_params: Initial parameter values
            callback: Optional callback function
            **kwargs: Additional arguments

        Returns:
            Optimization results
        """
        self.reset()
        self.start_time = time.time()

        # Create wrapper for objective function
        def wrapped_objective(params):
            value = objective_fn(params)

            # Update history
            current_time = time.time() - self.start_time
            self.history["params"].append(params.copy())
            self.history["values"].append(value)
            self.history["times"].append(current_time)

            # Update best value and parameters
            if value < self.best_value:
                self.best_value = value
                self.best_params = params.copy()

            self.iterations += 1

            # Call callback if provided
            if callback:
                callback(params)

            return value

        # Run the optimization based on the chosen method
        if self.method == "differential_evolution":
            # Set bounds for parameters (typically [-π, π] for quantum circuits)
            bounds = [(-np.pi, np.pi) for _ in range(len(initial_params))]

            result = differential_evolution(
                wrapped_objective,
                bounds,
                popsize=self.population_size,
                init="random",
                maxiter=kwargs.get("max_iterations", 100),
                seed=kwargs.get("seed", 42),
            )

            self.best_params = result.x
            self.best_value = result.fun
            self.converged = result.success

        elif self.method == "nelder-mead":
            result = minimize(
                wrapped_objective,
                initial_params,
                method="Nelder-Mead",
                options={
                    "maxiter": kwargs.get("max_iterations", 100),
                    "adaptive": True,
                },
            )

            self.best_params = result.x
            self.best_value = result.fun
            self.converged = result.success

        elif self.method == "basin-hopping":
            # Basin-hopping is good for finding global minima
            result = basinhopping(
                wrapped_objective,
                initial_params,
                niter=kwargs.get("max_iterations", 10),
                T=kwargs.get("temperature", 1.0),
                stepsize=kwargs.get("stepsize", 0.5),
                minimizer_kwargs={"method": "BFGS"},
            )

            self.best_params = result.x
            self.best_value = result.fun
            self.converged = result.success

        else:
            raise ValueError(f"Unknown optimization method: {self.method}")

        self.total_time = time.time() - self.start_time

        return {
            "params": self.best_params,
            "value": self.best_value,
            "iterations": self.iterations,
            "time": self.total_time,
            "history": self.history,
            "converged": self.converged,
        }


class AdaptiveOptimizer(QuantumOptimizer):
    """
    Adaptive optimizer that changes strategies during optimization.

    This optimizer switches between different methods based on progress
    and can dynamically adjust hyperparameters.
    """

    def __init__(
        self,
        methods: List[str] = None,
        phase_iterations: int = 50,
        learning_rate: float = 0.01,
    ):
        """
        Initialize the adaptive optimizer.

        Args:
            methods: List of optimization methods to use
            phase_iterations: Iterations per optimization phase
            learning_rate: Initial learning rate
        """
        super().__init__(name="Adaptive Optimizer")
        self.methods = methods or ["BFGS", "Nelder-Mead", "differential_evolution"]
        self.phase_iterations = phase_iterations
        self.learning_rate = learning_rate
        self.current_method_idx = 0
        self.phase_count = 0
        self.stagnation_count = 0

    def _select_next_method(self, progress_rate: float):
        """
        Select the next optimization method based on progress.

        Args:
            progress_rate: Rate of improvement in the objective
        """
        # If making good progress, stay with current method
        if progress_rate > 0.01:
            return

        # If stuck, switch methods
        self.current_method_idx = (self.current_method_idx + 1) % len(self.methods)
        self.phase_count += 1

        # Reduce learning rate with each phase
        self.learning_rate *= 0.8

    def minimize(
        self,
        objective_fn: Callable,
        initial_params: np.ndarray,
        callback: Optional[Callable] = None,
        **kwargs,
    ) -> Dict:
        """
        Minimize an objective function with adaptive strategy.

        Args:
            objective_fn: Function to minimize
            initial_params: Initial parameter values
            callback: Optional callback function
            **kwargs: Additional arguments

        Returns:
            Optimization results
        """
        self.reset()
        self.start_time = time.time()
        self.current_method_idx = 0
        self.phase_count = 0
        self.stagnation_count = 0

        # Initialize with the first method's parameters
        current_params = initial_params.copy()
        last_best_value = float("inf")

        # Set maximum total iterations
        max_total_iterations = kwargs.get("max_iterations", 200)
        max_phases = kwargs.get("max_phases", 5)

        # Main optimization loop across different methods
        while (
            self.iterations < max_total_iterations
            and self.phase_count < max_phases
            and not self.converged
        ):
            # Get current method
            current_method = self.methods[self.current_method_idx]

            # Create phase-specific objective function wrapper
            phase_best_value = float("inf")
            phase_iterations = 0

            def wrapped_objective(params):
                nonlocal phase_best_value, phase_iterations

                value = objective_fn(params)
                phase_iterations += 1

                # Update history
                current_time = time.time() - self.start_time
                self.history["params"].append(params.copy())
                self.history["values"].append(value)
                self.history["times"].append(current_time)

                # Update best values
                if value < phase_best_value:
                    phase_best_value = value

                if value < self.best_value:
                    self.best_value = value
                    self.best_params = params.copy()

                # Call original callback if provided
                if callback:
                    callback(params)

                return value

            # Configure the current phase optimization
            phase_options = {
                "maxiter": min(
                    self.phase_iterations, max_total_iterations - self.iterations
                )
            }

            # Add method-specific options
            if current_method == "BFGS":
                # BFGS with specific learning rate
                result = minimize(
                    wrapped_objective,
                    current_params,
                    method=current_method,
                    options=phase_options,
                )
                current_params = result.x

            elif current_method == "Nelder-Mead":
                # Nelder-Mead with adaptive simplex
                result = minimize(
                    wrapped_objective,
                    current_params,
                    method=current_method,
                    options={**phase_options, "adaptive": True},
                )
                current_params = result.x

            elif current_method == "differential_evolution":
                # Differential evolution needs bounds
                bounds = [(-np.pi, np.pi) for _ in range(len(current_params))]

                result = differential_evolution(
                    wrapped_objective,
                    bounds,
                    maxiter=phase_options["maxiter"] // 10,  # DE uses generations
                    popsize=min(15, 5 * len(current_params)),
                    init="latinhypercube",
                )
                current_params = result.x

            else:
                # Fall back to simple gradient descent for unknown methods
                simple_gd_iterations = phase_options["maxiter"]
                params = current_params.copy()
                lr = self.learning_rate

                for _ in range(simple_gd_iterations):
                    # Finite difference gradient approximation
                    grad = np.zeros_like(params)
                    value = wrapped_objective(params)

                    for i in range(len(params)):
                        delta = np.zeros_like(params)
                        delta[i] = 1e-4
                        value_plus = wrapped_objective(params + delta)
                        grad[i] = (value_plus - value) / 1e-4

                    # Update parameters
                    params -= lr * grad

                current_params = params
                # Phase success status tracked by result.success if needed

            # Update iterations count
            self.iterations += phase_iterations

            # Calculate progress and decide whether to switch methods
            progress_rate = (last_best_value - phase_best_value) / (
                abs(last_best_value) + 1e-10
            )
            self._select_next_method(progress_rate)

            # Check for convergence
            if abs(last_best_value - phase_best_value) < 1e-6:
                self.stagnation_count += 1
                if self.stagnation_count >= 2:
                    self.converged = True
            else:
                self.stagnation_count = 0

            last_best_value = phase_best_value

        self.total_time = time.time() - self.start_time

        return {
            "params": self.best_params,
            "value": self.best_value,
            "iterations": self.iterations,
            "time": self.total_time,
            "history": self.history,
            "converged": self.converged,
            "phases": self.phase_count,
        }


class ParallelTemperingOptimizer(QuantumOptimizer):
    """
    Optimizer that runs multiple optimization processes at different "temperatures".

    This approach is similar to parallel tempering in statistical physics and is
    effective for avoiding local minima.
    """

    def __init__(
        self, n_replicas: int = 4, base_optimizer: str = "BFGS", swap_interval: int = 10
    ):
        """
        Initialize the parallel tempering optimizer.

        Args:
            n_replicas: Number of parallel optimization processes
            base_optimizer: Base optimization method for each replica
            swap_interval: Interval for attempting swaps between replicas
        """
        super().__init__(name="Parallel Tempering")
        self.n_replicas = n_replicas
        self.base_optimizer = base_optimizer
        self.swap_interval = swap_interval
        self.replica_params = []
        self.replica_values = []
        self.temperatures = []
        self.accepted_swaps = 0
        self.total_swap_attempts = 0

    def _initialize_replicas(self, initial_params: np.ndarray):
        """
        Initialize the parameters and temperatures for all replicas.

        Args:
            initial_params: Initial parameter values
        """
        # Create slightly different initial parameters for each replica
        self.replica_params = []
        self.replica_values = [float("inf")] * self.n_replicas

        for i in range(self.n_replicas):
            # Add some noise to initial parameters for diversity
            noise_scale = 0.1 * (i + 1) / self.n_replicas
            noise = np.random.normal(0, noise_scale, size=initial_params.shape)
            self.replica_params.append(initial_params + noise)

        # Set up temperature ladder (higher temperature -> more exploration)
        self.temperatures = [1.0 * (1.5**i) for i in range(self.n_replicas)]

    def _attempt_swap(self, objective_fn: Callable):
        """
        Attempt to swap configurations between adjacent replicas.

        Args:
            objective_fn: Objective function
        """
        self.total_swap_attempts += 1

        # Randomly select a pair of adjacent replicas
        i = np.random.randint(0, self.n_replicas - 1)
        j = i + 1

        # Calculate acceptance probability using Metropolis criterion
        delta = (1.0 / self.temperatures[i] - 1.0 / self.temperatures[j]) * (
            self.replica_values[i] - self.replica_values[j]
        )

        # Accept or reject the swap
        if delta < 0 or np.random.random() < np.exp(-delta):
            # Swap parameters and values
            self.replica_params[i], self.replica_params[j] = (
                self.replica_params[j],
                self.replica_params[i],
            )
            self.replica_values[i], self.replica_values[j] = (
                self.replica_values[j],
                self.replica_values[i],
            )
            self.accepted_swaps += 1

    def minimize(
        self,
        objective_fn: Callable,
        initial_params: np.ndarray,
        callback: Optional[Callable] = None,
        **kwargs,
    ) -> Dict:
        """
        Minimize an objective function using parallel tempering.

        Args:
            objective_fn: Function to minimize
            initial_params: Initial parameter values
            callback: Optional callback function
            **kwargs: Additional arguments

        Returns:
            Optimization results
        """
        self.reset()
        self.start_time = time.time()

        # Initialize replicas
        self._initialize_replicas(initial_params)

        # Create temperature-dependent objective functions
        def create_replica_objective(replica_idx):
            def wrapped_objective(params):
                # Evaluate at original temperature
                value = objective_fn(params)

                # Store the true value
                self.replica_values[replica_idx] = value

                # For optimization purposes, scale by temperature
                # Higher temperature -> flatter landscape
                scaled_value = value / self.temperatures[replica_idx]

                # Update history (to store original unscaled value)
                current_time = time.time() - self.start_time
                self.history["params"].append(params.copy())
                self.history["values"].append(value)
                self.history["times"].append(current_time)

                # Update best value and parameters (using unscaled value)
                if value < self.best_value:
                    self.best_value = value
                    self.best_params = params.copy()

                self.iterations += 1

                # Call original callback if provided
                if callback and replica_idx == 0:  # Only call for the main replica
                    callback(params)

                return scaled_value

            return wrapped_objective

        # Main optimization loop
        max_iterations = kwargs.get("max_iterations", 100)
        iteration = 0

        while iteration < max_iterations and not self.converged:
            # Optimize each replica for a few steps
            for i in range(self.n_replicas):
                # Create replica-specific objective
                replica_obj = create_replica_objective(i)

                # Do a few steps of optimization
                steps = min(self.swap_interval, max_iterations - iteration)

                # Optimize this replica
                result = minimize(
                    replica_obj,
                    self.replica_params[i],
                    method=self.base_optimizer,
                    options={"maxiter": steps},
                )

                # Update parameters
                self.replica_params[i] = result.x

            # Attempt temperature swaps
            self._attempt_swap(objective_fn)

            # Update iteration count
            iteration += self.swap_interval

            # Check for convergence (if lowest temperature replica hasn't improved)
            if len(self.history["values"]) > 10:
                recent_values = [
                    v
                    for v, p in zip(
                        self.history["values"][-10:], self.history["params"][-10:]
                    )
                    if np.array_equal(p, self.replica_params[0])
                ]

                if recent_values and max(recent_values) - min(recent_values) < 1e-6:
                    self.converged = True

        self.total_time = time.time() - self.start_time

        return {
            "params": self.best_params,
            "value": self.best_value,
            "iterations": self.iterations,
            "time": self.total_time,
            "history": self.history,
            "converged": self.converged,
            "swap_acceptance_rate": self.accepted_swaps
            / max(1, self.total_swap_attempts),
        }


# Define a factory function to create optimizers


def create_optimizer(name: str, **kwargs) -> QuantumOptimizer:
    """
    Create an optimizer by name.

    Args:
        name: Name of the optimizer
        **kwargs: Additional arguments for the optimizer

    Returns:
        QuantumOptimizer instance
    """
    name = name.lower()

    if name in ["bfgs", "l-bfgs-b", "nelder-mead", "powell", "cg", "slsqp"]:
        # These are standard scipy optimizers, wrap them in NoiseAwareOptimizer
        return NoiseAwareOptimizer(base_optimizer=name.upper(), **kwargs)

    elif name == "noise_aware":
        base_optimizer = kwargs.pop("base_optimizer", "BFGS")
        return NoiseAwareOptimizer(base_optimizer=base_optimizer, **kwargs)

    elif name == "gradient_free":
        method = kwargs.pop("method", "differential_evolution")
        return GradientFreeOptimizer(method=method, **kwargs)

    elif name == "adaptive":
        return AdaptiveOptimizer(**kwargs)

    elif name == "parallel_tempering":
        return ParallelTemperingOptimizer(**kwargs)

    else:
        raise ValueError(f"Unknown optimizer: {name}")


# Example usage
if __name__ == "__main__":
    # Define a test objective function (Rosenbrock function)
    def rosenbrock(x):
        return sum(
            100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
            for i in range(len(x) - 1)
        )

    # Initial parameters
    initial_params = np.array([0.0, 0.0])

    # Try different optimizers
    optimizers = [
        create_optimizer("noise_aware", base_optimizer="BFGS"),
        create_optimizer("gradient_free", method="nelder-mead"),
        create_optimizer("adaptive"),
        create_optimizer("parallel_tempering", n_replicas=3),
    ]

    for opt in optimizers:
        print(f"\nTesting {opt.name}...")
        result = opt.minimize(rosenbrock, initial_params, max_iterations=50)
        print(f"Best value: {result['value']:.6f}")
        print(f"Best params: {result['params']}")
        print(f"Iterations: {result['iterations']}")
        print(f"Time: {result['time']:.2f} seconds")
