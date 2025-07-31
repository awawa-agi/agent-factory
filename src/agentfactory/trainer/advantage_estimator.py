"""
Advantage estimation for reinforcement learning training.

This module provides different advantage estimation strategies including:
- GRPO: Group Relative Policy Optimization  
- EMT-GRPO: Enhanced Multi-Turn Group Relative Policy Optimization
"""

import numpy as np
from typing import Dict, List

from ..hparams.algorithm_config import AlgorithmConfig
from ..rollout.core.config import SingleRolloutResult
from ..rollout.core.messages import AssistantMessage


class AdvantageEstimator:
    """Advantage estimator for RL training with support for multiple algorithms."""
    
    def __init__(self, config: AlgorithmConfig):
        self.config = config
        
    def add_advantages_to_results(self, rollout_results: Dict[str, List[SingleRolloutResult]]):
        """
        Main entry point: add advantages to rollout results based on configured algorithm.
        
        Args:
            rollout_results: Dict mapping prompt_id to list of rollout results
        """
        if self.config.adv_estimator == "grpo":
            self._add_advantages_grpo(rollout_results)
        elif self.config.adv_estimator == "emt_grpo":
            self._add_advantages_emt_grpo(rollout_results)
        else:
            raise ValueError(f"Unknown advantage estimator: {self.config.adv_estimator}")
    
    def _add_advantages_grpo(self, rollout_results: Dict[str, List[SingleRolloutResult]]):
        """
        Add GRPO advantages (traditional episode-level approach).
        Reuses existing logic from advantage_estumator.py.
        
        Args:
            rollout_results: Dict mapping prompt_id to list of rollout results
        """
        for prompt_id, results in rollout_results.items():
            # Extract valid returns (skip None values)
            valid_returns = [r.weighted_reward for r in results if r.weighted_reward is not None]
            
            if len(valid_returns) < self.config.min_group_size:
                # Set all results to None when insufficient group size
                for result in results:
                    result.advantage = None
                    for msg in result.messages.root:
                        if isinstance(msg, AssistantMessage):
                            msg.turn_level_advantage = None
                            msg.outcome_level_advantage = None
                            msg.emt_grpo_advantage = None
                continue
            
            # Calculate statistics from valid returns only
            returns_array = np.array(valid_returns, dtype=np.float64)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array, ddof=0)
            
            # Assign advantages
            for result in results:
                if result.weighted_reward is None:
                    result.advantage = None  # Keep None for invalid results
                    # Clear any existing advantages on assistant messages
                    for msg in result.messages.root:
                        if isinstance(msg, AssistantMessage):
                            msg.turn_level_advantage = None
                            msg.outcome_level_advantage = None
                            msg.emt_grpo_advantage = None
                else:
                    advantage = result.weighted_reward - mean_return
                    if self.config.norm_adv_by_std_in_grpo and std_return > 0:
                        advantage /= std_return
                    result.advantage = advantage
                    
                    # Set episode-level advantage on all assistant messages (GRPO mode)
                    for msg in result.messages.root:
                        if isinstance(msg, AssistantMessage):
                            msg.turn_level_advantage = None  # No turn-level in GRPO
                            msg.outcome_level_advantage = advantage  # Episode-level advantage
                            msg.emt_grpo_advantage = advantage  # Same as episode in GRPO mode
    
    def _add_advantages_emt_grpo(self, rollout_results: Dict[str, List[SingleRolloutResult]]):
        """
        Add EMT-GRPO advantages with turn-level and episode-level fusion.
        
        Args:
            rollout_results: Dict mapping prompt_id to list of rollout results
        """
        for prompt_id, results in rollout_results.items():
            # Filter valid results (those with episode rewards)
            valid_results = [r for r in results if r.weighted_reward is not None]
            
            if len(valid_results) < self.config.min_group_size:
                # Set all results to None and clear AssistantMessage attributes
                for result in results:
                    result.advantage = None
                    for msg in result.messages.root:
                        if isinstance(msg, AssistantMessage):
                            msg.turn_level_advantage = None
                            msg.outcome_level_advantage = None
                            msg.emt_grpo_advantage = None
                continue
            
            # Step 1: Calculate episode-level advantages (only for valid results)
            episode_advantages = self._calculate_episode_advantages(valid_results)
            
            # Step 2: Calculate turn-level advantages (only for valid results)
            turn_advantages = self._calculate_turn_advantages(valid_results)
            
            # Step 3: Apply EMT-GRPO formula to valid results
            self._apply_emt_grpo_formula(valid_results, episode_advantages, turn_advantages)
            
            # Step 4: Set invalid results to None
            valid_result_ids = {r.id for r in valid_results}
            for result in results:
                if result.id not in valid_result_ids:
                    result.advantage = None
                    # Clear any existing advantages on assistant messages
                    for msg in result.messages.root:
                        if isinstance(msg, AssistantMessage):
                            msg.turn_level_advantage = None
                            msg.outcome_level_advantage = None
                            msg.emt_grpo_advantage = None
    
    def _calculate_episode_advantages(self, results: List[SingleRolloutResult]) -> Dict[str, float]:
        """
        Calculate episode-level advantages from weighted rewards.
        Assumes all results have valid weighted_reward (not None).
        
        Args:
            results: List of valid rollout results
            
        Returns:
            Dict mapping result.id to episode advantage
        """
        episode_rewards = [r.weighted_reward for r in results]
        result_ids = [r.id for r in results]
        
        # Calculate standardized advantages using episode-specific normalization setting
        advantages = self._compute_group_advantages(
            episode_rewards, 
            normalize_by_std=self.config.norm_episode_adv_by_std
        )
        return dict(zip(result_ids, advantages))
    
    def _calculate_turn_advantages(self, results: List[SingleRolloutResult]) -> Dict[str, List[float]]:
        """
        Calculate turn-level advantages from assistant message turn rewards.
        
        Args:
            results: List of valid rollout results
            
        Returns:
            Dict mapping result.id to list of turn advantages
        """
        # Step 1: Extract all turn rewards across all results (group-wide standardization)
        all_turn_rewards = []
        turn_rewards_by_result = {}
        
        for result in results:
            result_turn_rewards = []
            for msg in result.messages.root:
                if isinstance(msg, AssistantMessage) and msg.weighted_turn_reward is not None:
                    reward_value = msg.weighted_turn_reward
                    all_turn_rewards.append(reward_value)
                    result_turn_rewards.append(reward_value)
            
            turn_rewards_by_result[result.id] = result_turn_rewards
        
        # Step 2: Group-wide standardization
        if not all_turn_rewards:
            return {result.id: [] for result in results}
        
        # Calculate standardized turn advantages using turn-specific normalization setting
        turn_advantages = {}
        for result_id, rewards in turn_rewards_by_result.items():
            if rewards:  # Only process if there are turn rewards for this result
                advantages = self._compute_group_advantages(
                    rewards, 
                    all_turn_rewards, 
                    normalize_by_std=self.config.norm_turn_adv_by_std
                )
                turn_advantages[result_id] = advantages
            else:
                turn_advantages[result_id] = []
        
        return turn_advantages
    
    def _apply_emt_grpo_formula(
        self, 
        results: List[SingleRolloutResult],
        episode_advantages: Dict[str, float],
        turn_advantages: Dict[str, List[float]]
    ):
        """
        Apply EMT-GRPO formula: A^MT_k = α × A^T_k + β × γ^(T-1-k) × A^O
        
        Args:
            results: List of valid rollout results to modify
            episode_advantages: Episode-level advantages by result.id
            turn_advantages: Turn-level advantages by result.id
        """
        alpha = self.config.emt_grpo_turn_weight
        beta = self.config.emt_grpo_episode_weight
        gamma = self.config.emt_grpo_gamma
        
        for result in results:
            episode_adv = episode_advantages.get(result.id, 0.0)
            turn_advs = turn_advantages.get(result.id, [])
            
            if not turn_advs:
                # Fallback to episode advantage if no turn advantages
                result.advantage = episode_adv
                
                # Set episode-level advantage on all assistant messages (fallback mode)
                for msg in result.messages.root:
                    if isinstance(msg, AssistantMessage):
                        msg.turn_level_advantage = None  # No turn-level available
                        msg.outcome_level_advantage = episode_adv  # Episode-level advantage
                        msg.emt_grpo_advantage = episode_adv  # Same as episode in fallback
                continue
            
            T = len(turn_advs)  # Total number of turns
            emt_advantages = []
            
            # Apply EMT-GRPO formula for each turn
            for t in range(T):
                turn_adv = turn_advs[t]
                decay_factor = gamma ** (T - 1 - t)  # Time discount: γ^(T-1-t)
                
                emt_advantage = alpha * turn_adv + beta * decay_factor * episode_adv
                emt_advantages.append(emt_advantage)
            
            # Store the aggregated advantage (weighted average by turn)
            result.advantage = np.mean(emt_advantages)
            
            # Store detailed advantages in metadata for debugging/analysis
            if result.metadata is None:
                result.metadata = {}
            result.metadata['emt_grpo_advantages'] = emt_advantages
            result.metadata['turn_advantages'] = turn_advs
            result.metadata['episode_advantage'] = episode_adv
            
            # Set advantages on individual AssistantMessage objects
            assistant_msg_index = 0
            for msg in result.messages.root:
                if isinstance(msg, AssistantMessage):
                    if assistant_msg_index < len(turn_advs):
                        # Set the individual advantages for this assistant message
                        msg.turn_level_advantage = turn_advs[assistant_msg_index]
                        msg.outcome_level_advantage = episode_adv
                        msg.emt_grpo_advantage = emt_advantages[assistant_msg_index]
                        assistant_msg_index += 1
                    else:
                        # Safety: clear any existing advantages for extra assistant messages
                        msg.turn_level_advantage = None
                        msg.outcome_level_advantage = None
                        msg.emt_grpo_advantage = None
    
    def _compute_group_advantages(self, rewards: List[float], reference_rewards: List[float] = None, normalize_by_std: bool = None) -> List[float]:
        """
        Compute normalized advantages for a group of rewards.
        
        Args:
            rewards: List of reward values to compute advantages for
            reference_rewards: Optional reference group for standardization (for turn-level)
            normalize_by_std: Override normalization setting. If None, uses config setting
            
        Returns:
            List of advantage values (same length as rewards)
        """
        if reference_rewards is None:
            reference_rewards = rewards
            
        # Use provided normalization setting or fall back to config
        if normalize_by_std is None:
            normalize_by_std = self.config.norm_adv_by_std_in_grpo
            
        # Calculate mean and std from reference group
        ref_array = np.array(reference_rewards, dtype=np.float64)
        mean_reward = np.mean(ref_array)
        std_reward = np.std(ref_array, ddof=0)
        
        # Compute advantages for target rewards
        advantages = []
        for reward in rewards:
            adv = reward - mean_reward
            if normalize_by_std and std_reward > 0:
                adv /= std_reward
            advantages.append(adv)
        
        return advantages