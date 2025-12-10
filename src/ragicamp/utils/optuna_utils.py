"""Optuna integration for hyperparameter optimization.

This module provides utilities for optimizing RAG system parameters
using Optuna's Bayesian optimization.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import optuna
    from optuna.study import Study
    from optuna.trial import Trial
    
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None
    Study = None
    Trial = None


class OptunaOptimizer:
    """Wrapper for Optuna hyperparameter optimization.
    
    This class provides a simplified interface for optimizing RAG system
    parameters like top_k, temperature, retrieval thresholds, etc.
    
    Example:
        >>> def objective(trial):
        ...     top_k = trial.suggest_int("top_k", 1, 20)
        ...     agent = FixedRAGAgent(model, retriever, top_k=top_k)
        ...     results = evaluator.evaluate(agent)
        ...     return results["f1"]
        >>> 
        >>> optimizer = OptunaOptimizer(
        ...     study_name="rag_optimization",
        ...     direction="maximize"
        ... )
        >>> best_params = optimizer.optimize(objective, n_trials=50)
    """
    
    def __init__(
        self,
        study_name: str = "ragicamp_study",
        direction: str = "maximize",
        storage: Optional[str] = None,
        load_if_exists: bool = True,
    ):
        """Initialize Optuna optimizer.
        
        Args:
            study_name: Name of the optimization study
            direction: "maximize" or "minimize"
            storage: Database URL for persistence (e.g., "sqlite:///optuna.db")
            load_if_exists: Load existing study if found
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "optuna not installed. Install with: pip install optuna"
            )
        
        self.study_name = study_name
        self.direction = direction
        self.storage = storage
        self.load_if_exists = load_if_exists
        self._study = None
    
    def create_study(self) -> Study:
        """Create or load an Optuna study."""
        self._study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            storage=self.storage,
            load_if_exists=self.load_if_exists,
        )
        return self._study
    
    def optimize(
        self,
        objective: Callable[[Trial], float],
        n_trials: int = 20,
        timeout: Optional[int] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, Any]:
        """Run optimization.
        
        Args:
            objective: Objective function taking a Trial and returning a score
            n_trials: Number of optimization trials
            timeout: Optional timeout in seconds
            callbacks: Optional list of callback functions
            
        Returns:
            Dictionary with best parameters and value
        """
        if self._study is None:
            self.create_study()
        
        print(f"🔍 Starting Optuna optimization: {self.study_name}")
        print(f"   Direction: {self.direction}")
        print(f"   Trials: {n_trials}")
        
        self._study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=callbacks,
            show_progress_bar=True,
        )
        
        print(f"\n✅ Optimization complete!")
        print(f"   Best value: {self._study.best_value:.4f}")
        print(f"   Best params: {self._study.best_params}")
        
        return {
            "best_params": self._study.best_params,
            "best_value": self._study.best_value,
            "best_trial": self._study.best_trial.number,
            "n_trials": len(self._study.trials),
        }
    
    def get_study(self) -> Optional[Study]:
        """Get the current study."""
        return self._study
    
    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Get best parameters from completed study."""
        if self._study is None:
            return None
        return self._study.best_params
    
    def get_best_value(self) -> Optional[float]:
        """Get best value from completed study."""
        if self._study is None:
            return None
        return self._study.best_value
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history.
        
        Args:
            save_path: Optional path to save plot
        """
        if self._study is None:
            print("⚠️  No study to plot")
            return
        
        try:
            from optuna.visualization import plot_optimization_history
            import plotly
            
            fig = plot_optimization_history(self._study)
            
            if save_path:
                fig.write_html(save_path)
                print(f"📊 Saved optimization history to {save_path}")
            else:
                fig.show()
        except ImportError:
            print("⚠️  plotly not installed. Install with: pip install plotly")
    
    def plot_param_importances(self, save_path: Optional[str] = None):
        """Plot parameter importances.
        
        Args:
            save_path: Optional path to save plot
        """
        if self._study is None:
            print("⚠️  No study to plot")
            return
        
        try:
            from optuna.visualization import plot_param_importances
            import plotly
            
            fig = plot_param_importances(self._study)
            
            if save_path:
                fig.write_html(save_path)
                print(f"📊 Saved param importances to {save_path}")
            else:
                fig.show()
        except ImportError:
            print("⚠️  plotly not installed. Install with: pip install plotly")


def create_rag_objective(
    agent_factory: Callable,
    evaluator_factory: Callable,
    metric_name: str = "f1",
    param_space: Optional[Dict[str, Any]] = None,
) -> Callable[[Trial], float]:
    """Create an objective function for RAG optimization.
    
    Args:
        agent_factory: Function(trial_params) -> agent
        evaluator_factory: Function(agent) -> evaluator
        metric_name: Name of metric to optimize
        param_space: Optional parameter space definition
        
    Returns:
        Objective function for Optuna
        
    Example:
        >>> def make_agent(params):
        ...     return FixedRAGAgent(
        ...         model, retriever,
        ...         top_k=params["top_k"],
        ...         max_tokens=params["max_tokens"]
        ...     )
        >>> 
        >>> def make_evaluator(agent):
        ...     return Evaluator(agent, dataset, metrics)
        >>> 
        >>> objective = create_rag_objective(
        ...     make_agent, make_evaluator, metric_name="f1"
        ... )
    """
    if param_space is None:
        # Default search space for RAG
        param_space = {
            "top_k": ("int", 1, 20),
            "temperature": ("float", 0.1, 2.0),
            "max_tokens": ("int", 50, 500),
        }
    
    def objective(trial: Trial) -> float:
        # Suggest parameters
        params = {}
        for param_name, param_config in param_space.items():
            param_type = param_config[0]
            
            if param_type == "int":
                params[param_name] = trial.suggest_int(param_name, param_config[1], param_config[2])
            elif param_type == "float":
                params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2])
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, param_config[1])
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
        
        # Create agent with suggested parameters
        agent = agent_factory(params)
        
        # Evaluate
        evaluator = evaluator_factory(agent)
        results = evaluator.evaluate()
        
        # Return metric to optimize
        if metric_name not in results:
            raise ValueError(f"Metric {metric_name} not found in results: {results.keys()}")
        
        return results[metric_name]
    
    return objective


def optimize_from_config(config: Dict[str, Any], objective: Callable) -> Dict[str, Any]:
    """Run optimization from experiment config.
    
    Args:
        config: Experiment configuration with optuna section
        objective: Objective function
        
    Returns:
        Best parameters and results
    """
    optuna_config = config.get("optuna", {})
    
    if not optuna_config.get("enabled", False):
        print("⚠️  Optuna optimization not enabled in config")
        return {}
    
    optimizer = OptunaOptimizer(
        study_name=optuna_config.get("study_name", "ragicamp_study"),
        direction=optuna_config.get("direction", "maximize"),
        storage=optuna_config.get("storage"),
    )
    
    results = optimizer.optimize(
        objective,
        n_trials=optuna_config.get("n_trials", 20),
    )
    
    return results
