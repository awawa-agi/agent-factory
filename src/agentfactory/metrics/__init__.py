"""
Metrics calculation and management for AgentFactory training.

This module provides a clean separation between metrics computation and logging,
allowing for efficient, testable, and extensible metric collection.
"""

from .metrics_manager import MetricsManager
from .statistics_utils import StatisticsUtils

__all__ = ["MetricsManager", "StatisticsUtils"]