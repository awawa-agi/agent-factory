"""
Conversation data adapter for the unified visualizer
Handles conversion of rollout results to conversation format
"""

from typing import List, Dict, Any, Optional
from ...rollout.core.config import SingleRolloutResult
from ...rollout.core.messages import Messages, MessageRole


class ConversationAdapter:
    """Adapter for converting rollout data to conversation format"""
    
    @staticmethod
    def adapt_rollout_result(result: SingleRolloutResult) -> Dict[str, Any]:
        """
        Convert a SingleRolloutResult to conversation format
        
        Args:
            result: SingleRolloutResult to convert
            
        Returns:
            Dictionary with conversation data
        """
        return {
            "id": result.id or "unknown",
            "is_success": result.is_success or False,
            "weighted_reward": result.weighted_reward or 0.0,
            "conversation": ConversationAdapter._extract_conversation(result.messages),
            "system_prompt": ConversationAdapter._extract_system_prompt(result.messages),
            "rewards": result.reward_components or {},
            "metadata": ConversationAdapter._extract_metadata(result),
            "advantage": result.advantage
        }
    
    @staticmethod
    def _extract_conversation(messages: Optional[Messages]) -> List[Dict[str, Any]]:
        """Extract conversation messages excluding system messages"""
        if not messages:
            return []
        
        conversation = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                continue
                
            msg_data = {
                "role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
                "content": msg.get_text(),
                "timestamp": "2025-07-05 14:30:00"  # Mock timestamp
            }
            
            # Add tool-specific data for tool messages
            if msg.role == 'tool':
                if hasattr(msg, 'tool_name') and msg.tool_name:
                    msg_data["tool_name"] = msg.tool_name
                if hasattr(msg, 'structured_output') and msg.structured_output:
                    msg_data["structured_output"] = msg.structured_output
            
            # Add turn-level rewards for assistant messages
            if hasattr(msg, 'turn_reward_components') and msg.turn_reward_components:
                msg_data["rewards"] = msg.turn_reward_components.copy()
                
                # Add additional reward fields if present
                if hasattr(msg, 'weighted_turn_reward') and msg.weighted_turn_reward is not None:
                    msg_data["rewards"]["Weighted Turn Reward"] = msg.weighted_turn_reward
                    
                if hasattr(msg, 'turn_level_advantage') and msg.turn_level_advantage is not None:
                    msg_data["rewards"]["Turn Advantage"] = msg.turn_level_advantage
                    
                if hasattr(msg, 'outcome_level_advantage') and msg.outcome_level_advantage is not None:
                    msg_data["rewards"]["Outcome Advantage"] = msg.outcome_level_advantage
                    
                if hasattr(msg, 'emt_grpo_advantage') and msg.emt_grpo_advantage is not None:
                    msg_data["rewards"]["EMT-GRPO Advantage"] = msg.emt_grpo_advantage
            
            conversation.append(msg_data)
        
        return conversation
    
    @staticmethod
    def _extract_system_prompt(messages: Optional[Messages]) -> Optional[str]:
        """Extract system prompt from messages"""
        if not messages:
            return None
            
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                return msg.get_text()
        
        return None
    
    @staticmethod
    def _extract_metadata(result: SingleRolloutResult) -> Dict[str, Any]:
        """Extract metadata from rollout result"""
        metadata = {}
        
        # Token statistics
        if result.token_stats:
            total_tokens = result.token_stats.total_tokens or 0
            completion_tokens = result.token_stats.num_completion_tokens or 0
            input_tokens = total_tokens - completion_tokens if total_tokens else 0
            
            metadata.update({
                "inputTokens": input_tokens,
                "outputTokens": completion_tokens,
                "totalTokens": total_tokens,
            })
        
        # Execution time
        if result.execution_time:
            metadata["executionTime"] = f"{result.execution_time:.1f}s"
        
        # Additional metadata from result
        if result.metadata:
            metadata.update(result.metadata)
        
        return metadata
    
    @staticmethod
    def adapt_multiple_results(results: List[SingleRolloutResult]) -> List[Dict[str, Any]]:
        """Adapt multiple rollout results"""
        return [ConversationAdapter.adapt_rollout_result(result) for result in results]
    
    @staticmethod
    def adapt_grouped_results(grouped_results: Dict[str, List[SingleRolloutResult]]) -> Dict[str, List[Dict[str, Any]]]:
        """Adapt grouped rollout results"""
        adapted = {}
        for group_id, results in grouped_results.items():
            adapted[group_id] = ConversationAdapter.adapt_multiple_results(results)
        return adapted