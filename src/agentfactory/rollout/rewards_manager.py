"""
Reward management system for evaluating agent performance.

This module provides:
1. A central registry for reward functions
2. Utilities for reward calculation and advantage computation
3. Integration with the rollout system
"""

from typing import Dict, List, Any, Tuple, Optional, Callable, Union, TypeVar, Hashable, TypedDict
import math
from loguru import logger

from . import Messages

# Type for request IDs - can be any hashable type
RequestIdType = TypeVar('RequestIdType', bound=Hashable)


class RewardFunctionEntry(TypedDict):
    """Registry entry for reward functions"""
    func: Callable[[Messages, Dict[str, Any]], Any]
    supports_episode: bool
    supports_process: bool

class StandardRewardReturn(TypedDict, total=False):
    """Standardized return format for reward functions"""
    episode: Union[float, Dict[str, float]]
    process: List[Dict[str, float]]

class EpisodeRewardResult(TypedDict):
    """Episode reward calculation result"""
    weighted_reward: float | None
    components: Dict[str, float]

class TurnRewardResult(TypedDict):
    """Turn reward calculation result"""
    components: List[Dict[str, float]]  # Per assistant message components
    weighted_rewards: List[float | None]  # Per assistant message weighted scores

class UnifiedRewardResult(TypedDict):
    """Unified reward calculation result"""
    episode: EpisodeRewardResult
    turn: TurnRewardResult


class RewardManager:
    """Unified manager for episode and turn reward functions."""
    
    # Registry of reward functions
    _registry: Dict[str, RewardFunctionEntry] = {}
    
    @classmethod
    def register(
        cls, 
        name: Optional[str] = None, 
        episode: bool = True, 
        process: bool = False
    ) -> Callable:
        """Decorator to register a reward function.
        
        Args:
            name: Optional custom name for the reward function.
            episode: Whether this function supports episode rewards.
            process: Whether this function supports turn rewards.
                 
        Returns:
            The decorator function.
        """
        def decorator(func: Callable) -> Callable:
            # Basic validation
            if not episode and not process:
                raise ValueError(f"Reward function must support at least episode or turn rewards")
            
            func_name = name or func.__name__
            cls._registry[func_name] = RewardFunctionEntry(
                func=func,
                supports_episode=episode,
                supports_process=process
            )
            return func
        return decorator
    
    @classmethod
    def register_turn(cls, name: Optional[str] = None) -> Callable:
        """Convenience decorator for turn-only reward functions."""
        return cls.register(name=name, episode=False, process=True)
    
    @classmethod
    def get_reward_function(cls, name: str) -> Optional[Callable]:
        """Get a reward function by name.
        
        Args:
            name: Name of the reward function to retrieve.
            
        Returns:
            The reward function if found, None otherwise.
        """
        entry = cls._registry.get(name)
        return entry['func'] if entry else None
    
    @classmethod
    def supports_episode(cls, name: str) -> bool:
        """Check if a reward function supports episode rewards."""
        entry = cls._registry.get(name)
        return entry['supports_episode'] if entry else False
    
    @classmethod
    def supports_turn(cls, name: str) -> bool:
        """Check if a reward function supports turn rewards."""
        entry = cls._registry.get(name)
        return entry['supports_process'] if entry else False
    
    @classmethod
    def list_available_rewards(cls) -> List[str]:
        """List all available reward function names.
        
        Returns:
            List of registered reward function names.
        """
        return list(cls._registry.keys())
    
    @classmethod
    def list_episode_rewards(cls) -> List[str]:
        """List reward functions that support episode rewards."""
        return [name for name, entry in cls._registry.items() if entry['supports_episode']]
    
    @classmethod
    def list_turn_rewards(cls) -> List[str]:
        """List reward functions that support turn rewards."""
        return [name for name, entry in cls._registry.items() if entry['supports_process']]
    
    @classmethod
    def calculate_rewards(
        cls, 
        messages: Messages,
        metadata: Dict[str, Any],
        reward_metrics: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate episode rewards based on specified metrics (legacy method).
        
        Args:
            messages: Conversation messages.
            metadata: Metadata dictionary for reward functions.
            reward_metrics: Dictionary mapping reward function names to weights.
            
        Returns:
            Tuple of (total weighted reward, component rewards dictionary).
        """
        # Use the unified method but only return episode results
        result = cls.calculate_unified_rewards(
            messages, metadata, reward_metrics, {}
        )
        return result['episode']['weighted_reward'], result['episode']['components']
        
    @classmethod
    def calculate_unified_rewards(
        cls,
        messages: Messages,
        metadata: Dict[str, Any],
        episode_reward_metrics: Dict[str, float],
        turn_reward_metrics: Dict[str, float]
    ) -> UnifiedRewardResult:
        """Calculate both episode and turn rewards.
        
        Args:
            messages: Conversation messages.
            metadata: Metadata dictionary for reward functions.
            episode_reward_metrics: Dict mapping episode reward names to weights.
            turn_reward_metrics: Dict mapping turn reward names to weights.
            
        Returns:
            UnifiedRewardResult containing both episode and turn rewards.
        """
        # Collect all function names needed
        all_function_names = set()
        all_function_names.update(metric.split('.')[0] for metric in episode_reward_metrics.keys())
        all_function_names.update(metric.split('.')[0] for metric in turn_reward_metrics.keys())
        
        num_assistants = sum(1 for msg in messages.root if msg.role == "assistant")
        
        # Execute and validate all functions
        validated_results = {}
        for func_name in all_function_names:
            entry = cls._registry.get(func_name)
            if not entry:
                logger.error(f"Reward function '{func_name}' not found in registry")
                continue
                
            try:
                raw_result = entry['func'](messages, metadata)
                validated_results[func_name] = cls._validate_and_normalize_result(
                    raw_result, 
                    entry['supports_episode'], 
                    entry['supports_process'], 
                    func_name,
                    num_assistants
                )
            except Exception as e:
                logger.error(f"Error in reward function '{func_name}': {e}")
                continue
        
        # Build final results
        episode_result = cls._build_episode_results(validated_results, episode_reward_metrics)
        turn_result = cls._build_turn_results(validated_results, turn_reward_metrics, num_assistants)
        
        return UnifiedRewardResult(episode=episode_result, turn=turn_result)
    
    @classmethod
    def _validate_and_normalize_result(
        cls, 
        raw_result: Any, 
        supports_episode: bool, 
        supports_process: bool, 
        func_name: str,
        expected_assistant_count: int
    ) -> StandardRewardReturn:
        """Validate and normalize reward function result based on registration."""
        
        normalized = StandardRewardReturn()
        
        # Case 1: Episode-only support
        if supports_episode and not supports_process:
            if isinstance(raw_result, (int, float)):
                normalized['episode'] = float(raw_result)
            elif isinstance(raw_result, dict) and 'episode' not in raw_result and 'process' not in raw_result:
                # Pure dict, treat as episode components
                normalized['episode'] = raw_result
            elif isinstance(raw_result, dict) and 'episode' in raw_result:
                normalized['episode'] = raw_result['episode']
            else:
                raise ValueError(f"Function {func_name} registered as episode-only but returned invalid format: {type(raw_result)}")
        
        # Case 2: Turn-only support
        elif supports_process and not supports_episode:
            if isinstance(raw_result, list):
                cls._validate_turn_list(raw_result, func_name, expected_assistant_count)
                normalized['process'] = raw_result
            elif isinstance(raw_result, dict) and 'process' in raw_result:
                cls._validate_turn_list(raw_result['process'], func_name, expected_assistant_count)
                normalized['process'] = raw_result['process']
            else:
                raise ValueError(f"Function {func_name} registered as turn-only but returned invalid format: {type(raw_result)}")
        
        # Case 3: Both episode and turn support
        elif supports_episode and supports_process:
            if isinstance(raw_result, dict) and 'episode' in raw_result and 'process' in raw_result:
                # Standard format
                normalized['episode'] = raw_result['episode']
                normalized['process'] = raw_result['process']
                cls._validate_turn_list(raw_result['process'], func_name, expected_assistant_count)
            else:
                raise ValueError(
                    f"Function {func_name} registered as supporting both episode and turn "
                    f"but must return dict with both 'episode' and 'process' keys"
                )
        
        return normalized
    
    @classmethod
    def _validate_turn_list(cls, turn_data: Any, func_name: str, expected_count: int) -> None:
        """Validate turn reward list format and consistency."""
        if not isinstance(turn_data, list):
            raise ValueError(f"Turn data from {func_name} must be a list")
        
        if len(turn_data) != expected_count:
            raise ValueError(
                f"Turn rewards from {func_name} has {len(turn_data)} items, "
                f"expected {expected_count} assistant messages"
            )
        
        if not turn_data:
            return
        
        # Check each step is dict and keys are consistent
        first_keys = None
        for i, step_reward in enumerate(turn_data):
            if not isinstance(step_reward, dict):
                raise ValueError(f"Turn reward step {i} from {func_name} must be dict")
            
            step_keys = set(step_reward.keys())
            if first_keys is None:
                first_keys = step_keys
            elif step_keys != first_keys:
                raise ValueError(
                    f"Inconsistent keys in {func_name}: step 0 has {first_keys}, "
                    f"step {i} has {step_keys}"
                )
    
    @classmethod
    def _build_episode_results(
        cls, 
        validated_results: Dict[str, StandardRewardReturn], 
        episode_metrics: Dict[str, float]
    ) -> EpisodeRewardResult:
        """Build episode reward results from validated function outputs."""
        components = {}
        weighted_total = 0.0
        has_rewards = False
        
        for metric_name, weight in episode_metrics.items():
            reward_name = metric_name.split('.')[0]
            
            if reward_name in validated_results:
                result = validated_results[reward_name]
                episode_data = result.get('episode')
                
                if episode_data is not None:
                    value = cls._extract_metric_value(episode_data, metric_name)
                    if value is not None:
                        components[metric_name] = value
                        weighted_total += value * weight
                        has_rewards = True
        
        return EpisodeRewardResult(
            weighted_reward=weighted_total if has_rewards else None, 
            components=components
        )
    
    @classmethod
    def _build_turn_results(
        cls,
        validated_results: Dict[str, StandardRewardReturn],
        turn_metrics: Dict[str, float],
        num_assistants: int
    ) -> TurnRewardResult:
        """Build turn reward results from validated function outputs."""
        components = [{} for _ in range(num_assistants)] if num_assistants > 0 else []
        weighted_rewards = [0.0 for _ in range(num_assistants)] if num_assistants > 0 else []
        has_turn_rewards = [False for _ in range(num_assistants)] if num_assistants > 0 else []
        
        for metric_name, weight in turn_metrics.items():
            reward_name = metric_name.split('.')[0]
            
            if reward_name in validated_results:
                result = validated_results[reward_name]
                turn_data = result.get('process')
                
                if turn_data and len(turn_data) == num_assistants:
                    for i, step_rewards in enumerate(turn_data):
                        value = cls._extract_metric_value(step_rewards, metric_name)
                        if value is not None:
                            components[i][metric_name] = value
                            weighted_rewards[i] += value * weight
                            has_turn_rewards[i] = True
        
        # Convert 0.0 to None for assistants with no turn rewards
        final_weighted_rewards = [
            reward if has_reward else None 
            for reward, has_reward in zip(weighted_rewards, has_turn_rewards)
        ]
        
        return TurnRewardResult(components=components, weighted_rewards=final_weighted_rewards)
    
    @classmethod
    def _extract_metric_value(cls, data: Union[float, Dict[str, float]], metric_name: str) -> Optional[float]:
        """Extract specific metric value from reward data."""
        if isinstance(data, (int, float)):
            # For simple metrics like "reward_name" (not "reward_name.component")
            if '.' not in metric_name.split('.', 1)[1] if '.' in metric_name else True:
                return float(data)
        elif isinstance(data, dict):
            # For component metrics like "reward_name.component"
            if '.' in metric_name:
                component = metric_name.split('.', 1)[1]
                return data.get(component)
            else:
                # If metric_name has no component, this shouldn't happen with dict data
                logger.warning(f"Metric {metric_name} expects single value but got dict")
        
        return None