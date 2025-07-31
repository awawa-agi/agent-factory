"""
Utility functions for statistical computations in metrics.

This module provides optimized, vectorized statistical operations to avoid
repeated computation patterns and improve performance.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union


class StatisticsUtils:
    """Utilities for efficient statistical computations."""
    
    @staticmethod
    def compute_stats(
        values: Union[List[float], np.ndarray], 
        prefix: str, 
        stats: List[str] = ["mean", "min", "max", "std"]
    ) -> Dict[str, float]:
        """
        Compute specified statistics for numeric values efficiently.
        
        Args:
            values: Input numeric values (list or numpy array)
            prefix: Prefix for metric keys (e.g., "tokens/completion")
            stats: List of statistics to compute. Options: ["mean", "min", "max", "std", "sum", "count"]
            
        Returns:
            Dictionary with requested statistics
        """
        if not values:
            return {}
            
        values_array = np.asarray(values)
        if len(values_array) == 0:
            return {}
        
        metrics = {}
        
        # Compute requested statistics
        if "mean" in stats:
            metrics[f"{prefix}/mean"] = float(np.mean(values_array))
        if "min" in stats:
            metrics[f"{prefix}/min"] = float(np.min(values_array))
        if "max" in stats:
            metrics[f"{prefix}/max"] = float(np.max(values_array))
        if "std" in stats:
            metrics[f"{prefix}/std"] = float(np.std(values_array))
        if "sum" in stats:
            metrics[f"{prefix}/sum"] = float(np.sum(values_array))
        if "count" in stats:
            metrics[f"{prefix}/count"] = len(values_array)
            
        return metrics
    
    @staticmethod
    def extract_prompt_id_vectorized(rollout_ids: List[str]) -> List[str]:
        """
        Batch extract prompt IDs from rollout IDs efficiently.
        
        Expected format: "prompt{i}_gen{j}" -> "prompt{i}"
        
        Args:
            rollout_ids: List of rollout ID strings
            
        Returns:
            List of extracted prompt IDs
        """
        return [
            rollout_id.rsplit("_", 1)[0] if "_" in rollout_id else rollout_id 
            for rollout_id in rollout_ids
        ]
    
    @staticmethod
    def group_values_by_key(items: List[Any], key_func, value_func) -> Dict[str, List[Any]]:
        """
        Group items by key and extract values efficiently.
        
        Args:
            items: List of items to group
            key_func: Function to extract grouping key from item
            value_func: Function to extract value from item
            
        Returns:
            Dictionary mapping keys to lists of values
        """
        groups = {}
        for item in items:
            key = key_func(item)
            value = value_func(item)
            if value is not None:  # Skip None values
                groups.setdefault(key, []).append(value)
        return groups
    
    @staticmethod
    def compute_component_stats(
        components_dict: Dict[str, List[float]], 
        prefix: str,
        stats: List[str] = ["mean", "std"]
    ) -> Dict[str, float]:
        """
        Compute statistics for multiple component groups.
        
        Args:
            components_dict: Dictionary mapping component names to value lists
            prefix: Prefix for metric keys (e.g., "rewards", "turn_rewards")
            stats: List of statistics to compute for each component
            
        Returns:
            Dictionary with requested statistics for each component
        """
        metrics = {}
        for name, values in components_dict.items():
            if values:
                component_metrics = StatisticsUtils.compute_stats(
                    values, f"{prefix}/{name}", stats
                )
                metrics.update(component_metrics)
        return metrics
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with default fallback."""
        return numerator / denominator if denominator > 0 else default