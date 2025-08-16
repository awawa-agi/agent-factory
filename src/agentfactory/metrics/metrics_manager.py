"""
Clean metrics computation with logical separation by data type.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from .statistics_utils import StatisticsUtils


class MetricsManager:
    """Clean metrics manager with logical function separation."""
    
    def __init__(self):
        self.stats = StatisticsUtils()
    
    def compute_rollout_metrics(self, rollout_results) -> Dict[str, Any]:
        """Compute all rollout metrics with clear separation by data type."""
        if not rollout_results:
            return {}
        
        metrics = {}
        metrics.update(self._compute_token_metrics(rollout_results))
        metrics.update(self._compute_success_metrics(rollout_results))
        metrics.update(self._compute_conversation_metrics(rollout_results))
        metrics.update(self._compute_tool_metrics(rollout_results))
        metrics.update(self._compute_reward_metrics(rollout_results))
        metrics.update(self._compute_efficiency_metrics(rollout_results))
        
        return metrics
    
    def compute_assistant_token_metrics(self, per_token_logs: List[Dict]) -> Dict[str, float]:
        """Compute assistant token statistics efficiently."""
        if not per_token_logs:
            return {}
        
        all_logps, all_entropies, all_advantages = [], [], []
        total_assistant_tokens = 0
        total_tokens = 0
        
        for log_entry in per_token_logs:
            masks = log_entry.get('assistant_masks')
            if masks is None:
                continue
                
            total_tokens += len(masks)
            assistant_indices = np.asarray(masks) == 1
            assistant_count = np.sum(assistant_indices)
            total_assistant_tokens += assistant_count
            
            if assistant_count > 0:
                for field, collector in [('logps', all_logps), ('entropies', all_entropies), ('advantages', all_advantages)]:
                    values = log_entry.get(field)
                    if values is not None:
                        collector.extend(np.asarray(values)[assistant_indices])
        
        metrics = {
            'tokens/assistant_count': total_assistant_tokens,
            'tokens/assistant_ratio': self.stats.safe_divide(total_assistant_tokens, total_tokens)
        }
        
        for name, values in [('tokens/assistant_logps', all_logps), ('tokens/assistant_entropies', all_entropies), ('tokens/assistant_advantages', all_advantages)]:
            if values:
                metrics.update(self.stats.compute_stats(values, name, ['mean', 'std']))
        
        return metrics
    
    def compute_all_metrics(self, rollout_results, per_token_logs: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Compute all metrics efficiently."""
        metrics = self.compute_rollout_metrics(rollout_results)
        if per_token_logs:
            metrics.update(self.compute_assistant_token_metrics(per_token_logs))
        return metrics
    
    def prepare_prompt_reward_histogram_data(self, rollout_results) -> List[float]:
        """Extract per-prompt average rewards for histogram."""
        if not rollout_results:
            return []
        
        prompt_rewards = {}
        def _extract_prompt_id_from_rollout_id(rollout_id: str) -> str:
            """Extract prompt ID from rollout ID. Expected format: prompt{i_prompt}_gen{i_gen}"""
            if "_gen" in rollout_id:
                return rollout_id.rsplit("_gen", 1)[0]
            return rollout_id
        
        for result in rollout_results:
            prompt_id = _extract_prompt_id_from_rollout_id(result.id)
            if result.weighted_reward is not None:
                prompt_rewards.setdefault(prompt_id, []).append(result.weighted_reward)
        
        return [np.mean(rewards) for rewards in prompt_rewards.values() if rewards]
    
    def _compute_token_metrics(self, rollout_results) -> Dict[str, Any]:
        """Process token-related statistics."""
        completion_tokens = []
        total_tokens = []
        
        for result in rollout_results:
            if result.token_stats:
                if result.token_stats.num_completion_tokens is not None:
                    completion_tokens.append(result.token_stats.num_completion_tokens)
                if result.token_stats.total_tokens is not None:
                    total_tokens.append(result.token_stats.total_tokens)
        
        metrics = {}
        if completion_tokens:
            metrics.update(self.stats.compute_stats(completion_tokens, "tokens/completion"))
        if total_tokens:
            metrics.update(self.stats.compute_stats(total_tokens, "tokens/total"))
        
        return metrics
    
    def _compute_success_metrics(self, rollout_results) -> Dict[str, Any]:
        """Process success rate statistics."""
        success_count = sum(1 for result in rollout_results if result.is_success)
        return {"success_rate": success_count / len(rollout_results)}
    
    def _compute_conversation_metrics(self, rollout_results) -> Dict[str, Any]:
        """Process conversation-related statistics."""
        conversation_rounds = [sum(1 for msg in r.messages if msg.role == 'assistant') 
                              for r in rollout_results]
        
        if conversation_rounds:
            return self.stats.compute_stats(conversation_rounds, "conversation/rounds", ["mean", "min", "max"])
        return {}
    
    def _compute_tool_metrics(self, rollout_results) -> Dict[str, Any]:
        """Process tool usage statistics."""
        tool_usage = {}
        total_tool_calls = []
        
        for result in rollout_results:
            rollout_tool_calls = 0
            if result.messages:
                for msg in result.messages:
                    if msg.role == "tool" and hasattr(msg, 'tool_name') and msg.tool_name:
                        rollout_tool_calls += 1
                        tool_usage[msg.tool_name] = tool_usage.get(msg.tool_name, 0) + 1
            total_tool_calls.append(rollout_tool_calls)
        
        metrics = {}
        if total_tool_calls:
            metrics.update(self.stats.compute_stats(total_tool_calls, "tools/total", ["mean", "min", "max"]))
        
        for tool_name, count in tool_usage.items():
            metrics[f"tools/{tool_name}/count"] = count
            metrics[f"tools/{tool_name}/avg_per_rollout"] = count / len(rollout_results)
        
        return metrics
    
    def _compute_reward_metrics(self, rollout_results) -> Dict[str, Any]:
        """Process reward and turn-level reward statistics."""
        reward_components = {}
        weighted_rewards = []
        turn_reward_components = {}
        weighted_turn_rewards = []
        
        for result in rollout_results:
            # Result-level rewards
            if result.reward_components and result.id:
                for name, value in result.reward_components.items():
                    reward_components.setdefault(name, []).append(value)
            
            if result.weighted_reward is not None:
                weighted_rewards.append(result.weighted_reward)
            
            # Turn-level rewards from messages
            if result.messages:
                for msg in result.messages:
                    if hasattr(msg, 'turn_reward_components') and msg.turn_reward_components:
                        for name, value in msg.turn_reward_components.items():
                            turn_reward_components.setdefault(name, []).append(value)
                    
                    if hasattr(msg, 'weighted_turn_reward') and msg.weighted_turn_reward is not None:
                        weighted_turn_rewards.append(msg.weighted_turn_reward)
        
        metrics = {}
        
        # Result-level reward metrics
        metrics.update(self.stats.compute_component_stats(reward_components, "rewards"))
        if weighted_rewards:
            metrics.update(self.stats.compute_stats(weighted_rewards, "rewards/weighted", ["mean", "std"]))
        
        # Turn-level reward metrics
        metrics.update(self.stats.compute_component_stats(turn_reward_components, "turn_rewards", ["mean", "std"]))
        if weighted_turn_rewards:
            metrics.update(self.stats.compute_stats(weighted_turn_rewards, "turn_rewards/weighted", ["mean", "std"]))
        
        return metrics
    
    def _compute_efficiency_metrics(self, rollout_results) -> Dict[str, Any]:
        """Process prompt efficiency statistics."""
        all_prompts = set()
        efficient_prompts = set()
        
        def _extract_prompt_id_from_rollout_id(rollout_id: str) -> str:
            """Extract prompt ID from rollout ID. Expected format: prompt{i_prompt}_gen{i_gen}"""
            if "_gen" in rollout_id:
                return rollout_id.rsplit("_gen", 1)[0]
            return rollout_id
        
        for result in rollout_results:
            prompt_id = _extract_prompt_id_from_rollout_id(result.id)
            all_prompts.add(prompt_id)
            if result.advantage and result.advantage != 0:
                efficient_prompts.add(prompt_id)
        
        if all_prompts:
            return {
                "efficiency/prompt_ratio": len(efficient_prompts) / len(all_prompts),
                "efficiency/efficient_count": len(efficient_prompts),
                "efficiency/total_count": len(all_prompts)
            }
        return {}