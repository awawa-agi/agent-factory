"""
Reward functions for evaluating agent performance.

This module contains reward functions that evaluate different aspects of agent behavior.
Each function is registered with the RewardManager for use in rollout evaluations.
"""

import re
import math
from typing import Dict, List, Any, Tuple, Optional, Union

from .rewards_manager import RewardManager
from .core.messages import Messages

# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions 
# ─────────────────────────────────────────────────────────────────────────────

# @RewardManager.register(name='alfworld', episode=True, process=True)
# def alfworld_reward(messages, metadata):
#     """
#     Reward function for the AlfWorld game.
    
#     Episode rewards: stage_passed, has_valid_action, invalid_command_count
#     Turn rewards: thinking_format, tool_format
#     """
#     # Episode-level rewards
#     episode_rewards = {
#         "stage_passed": 0,
#         "has_valid_action": 0,
#         "invalid_command_count": 0,
#     }
    
#     for msg in messages:
#         if msg.role == "tool":
#             output = msg.structured_output
#             is_valid = output.get('is_action_valid', False)
            
#             if is_valid:
#                 episode_rewards["has_valid_action"] = 1
#             else:
#                 episode_rewards["invalid_command_count"] += 1
            
#             if output.get('won', False):
#                 episode_rewards["stage_passed"] = 1
    
#     # Turn-level rewards for each assistant message
#     turn_rewards = []
#     message_list = messages.root
    
#     for i, msg in enumerate(message_list):
#         if msg.role == "assistant":
#             step_reward = {
#                 "thinking_format": 0.0,
#                 "tool_format": 0.0,
#             }
            
#             # Check thinking format: exactly one <think> and one </think>
#             text = msg.get_text()
#             think_start_count = text.count('<think>')
#             think_end_count = text.count('</think>')
            
#             if think_start_count == 1 and think_end_count == 1:
#                 step_reward["thinking_format"] = 1.0
            
#             # Check tool format: next message is tool with valid action
#             if i + 1 < len(message_list):
#                 next_msg = message_list[i + 1]
#                 if next_msg.role == "tool":
#                     output = next_msg.structured_output
#                     if output and output.get('is_action_valid', False):
#                         step_reward["tool_format"] = 1.0
            
#             turn_rewards.append(step_reward)
    
#     return {
#         "episode": episode_rewards,
#         "process": turn_rewards
#     }