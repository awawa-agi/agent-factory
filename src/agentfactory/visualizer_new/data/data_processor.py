"""
Unified data processor for the AgentFactory visualizer
Handles conversion of various data sources into a unified format
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

from ...rollout.core.config import SingleRolloutResult, TokenStats
from ...rollout.core.messages import Messages  
from .token_adapter import TokenAdapter
from .types import (
    TokenDataInput, TokenSequenceInput, CompressedTokenGroup,
    TokenGroupStats, TokenSequenceInfo
)
from .token_compressor import TokenDataCompressor


@dataclass
class AppData:
    """Unified data structure for the visualizer app"""
    groups: Dict[str, 'GroupData']
    view_type: str = "conversation"  # "conversation", "token", "unified"
    navigation_data: Optional[Dict[str, Any]] = None
    
    
@dataclass
class GroupData:
    """Data for a single group of results"""
    group_id: str
    stats: 'GroupStats'
    samples: List['SampleData']
    

@dataclass  
class GroupStats:
    """Statistics for a group"""
    total_results: int
    success_rate: str
    avg_reward: float
    execution_time: Optional[str] = None
    

@dataclass
class SampleData:
    """Data for a single sample (rollout result or token sequence)"""
    sample_id: str
    name: str
    is_success: bool
    weighted_reward: float
    conversation: Optional[List[Dict[str, Any]]] = None
    token_data: Optional[Dict[str, Any]] = None
    rewards: Optional[Dict[str, float]] = None
    system_prompt: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class UnifiedDataProcessor:
    """Processes various data sources into unified format"""
    
    def __init__(self, chunk_size: int = 500):
        self.token_adapter = TokenAdapter(chunk_size=chunk_size)
    
    def process_rollout_results(self, results: Union[List[SingleRolloutResult], Dict[str, List[SingleRolloutResult]]]) -> AppData:
        """
        Process rollout results into unified format
        
        Args:
            results: Either list of results or dict mapping group_id -> list of results
        """
        # Convert to grouped format if needed
        if isinstance(results, list):
            grouped_results = {"default_group": results}
        else:
            grouped_results = results
            
        groups = {}
        for group_id, group_results in grouped_results.items():
            groups[group_id] = self._process_group_results(group_id, group_results)
            
        return AppData(
            groups=groups,
            view_type="conversation",
            navigation_data=self._generate_navigation_data(groups)
        )
    
    def process_token_data(self, token_data: TokenDataInput) -> AppData:
        """
        Process token data into unified format with group support
        
        Args:
            token_data: Either list of sequences (auto-grouped as 'default_group') or dict mapping group_id -> sequences
        """
        # Convert to grouped format if needed
        if isinstance(token_data, list):
            grouped_token_data = {"default_group": token_data}
        else:
            grouped_token_data = token_data
            
        groups = {}
        for group_id, group_sequences in grouped_token_data.items():
            # Convert to standardized input format if needed
            standardized_sequences = []
            for seq in group_sequences:
                if isinstance(seq, dict) and 'tokens' in seq:
                    standardized_sequences.append(seq)
                else:
                    # Handle legacy formats through token adapter
                    adapted = self.token_adapter.adapt_legacy_format([seq])
                    if adapted:
                        standardized_sequences.append(adapted[0])
            
            # Create compressed group
            compressed_group = TokenDataCompressor.compress_group(group_id, standardized_sequences)
            groups[group_id] = self._create_group_data_from_compressed(compressed_group)
        
        return AppData(
            groups=groups,
            view_type="token",
            navigation_data=self._generate_navigation_data(groups)
        )
    
    def process_multiple_data_sources(self, 
                                    rollout_results: Optional[Union[List, Dict]] = None,
                                    token_data: Optional[List[Dict]] = None) -> AppData:
        """Process multiple data sources for unified view"""
        groups = {}
        
        if rollout_results:
            conversation_data = self.process_rollout_results(rollout_results)
            groups.update(conversation_data.groups)
            
        if token_data:
            token_app_data = self.process_token_data(token_data)
            groups.update(token_app_data.groups)
            
        return AppData(
            groups=groups,
            view_type="unified",
            navigation_data=self._generate_navigation_data(groups)
        )
    
    def _process_group_results(self, group_id: str, results: List[SingleRolloutResult]) -> GroupData:
        """Process a single group of rollout results"""
        if not results:
            return GroupData(
                group_id=group_id,
                stats=GroupStats(0, "0/0 (0.0%)", 0.0),
                samples=[]
            )
            
        # Calculate group statistics
        success_count = sum(1 for r in results if r.is_success)
        total_count = len(results)
        success_rate = f"{success_count}/{total_count} ({success_count/total_count*100:.1f}%)"
        avg_reward = sum(r.weighted_reward or 0 for r in results) / len(results)
        
        # Process individual samples
        samples = []
        for i, result in enumerate(results):
            sample = self._convert_rollout_to_sample(result, i)
            samples.append(sample)
            
        stats = GroupStats(
            total_results=total_count,
            success_rate=success_rate,
            avg_reward=avg_reward,
            execution_time=f"{sum(r.execution_time or 0 for r in results):.1f}s"
        )
        
        return GroupData(
            group_id=group_id,
            stats=stats,
            samples=samples
        )
    
    def _process_token_group(self, token_data: List[Dict[str, Any]]) -> GroupData:
        """Process token data into group format"""
        samples = []
        for i, seq_data in enumerate(token_data):
            sample = SampleData(
                sample_id=f"sequence_{i}",
                name=seq_data.get("display_id", f"Sequence {i+1}"),
                is_success=True,  # Token sequences don't have success/failure
                weighted_reward=0.0,
                token_data=seq_data,
                metadata={
                    "total_tokens": len(seq_data.get("tokens", [])),
                    "avg_logprob": sum(seq_data.get("logprobs", [])) / len(seq_data.get("logprobs", [1])),
                    "avg_entropy": sum(seq_data.get("entropies", [])) / len(seq_data.get("entropies", [1]))
                }
            )
            samples.append(sample)
            
        stats = GroupStats(
            total_results=len(samples),
            success_rate=f"{len(samples)}/{len(samples)} (100.0%)",
            avg_reward=0.0
        )
        
        return GroupData(
            group_id="token_sequences",
            stats=stats,
            samples=samples
        )
    
    def _process_optimized_token_group(self, optimized_data: Dict[str, Any]) -> GroupData:
        """Process optimized token data into group format"""
        samples = []
        sequences = optimized_data.get("sequences", [])
        
        for seq in sequences:
            # Create sample with optimized token data
            sample = SampleData(
                sample_id=seq["sequence_id"],
                name=seq["display_id"],
                is_success=True,  # Token sequences don't have success/failure
                weighted_reward=0.0,
                token_data=optimized_data,  # Pass entire optimized structure
                metadata={
                    "total_tokens": seq["length"],
                    "total_chunks": seq["total_chunks"],
                    "chunk_size": seq["chunk_size"],
                    "avg_logprob": seq["metadata"]["avg_logprob"],
                    "avg_entropy": seq["metadata"]["avg_entropy"],
                    "avg_advantage": seq["metadata"]["avg_advantage"],
                    "format_version": optimized_data["format_version"]
                }
            )
            samples.append(sample)
        
        # Calculate group stats
        total_tokens = sum(seq["length"] for seq in sequences)
        total_sequences = len(sequences)
        
        stats = GroupStats(
            total_results=total_sequences,
            success_rate=f"{total_sequences}/{total_sequences} (100.0%)",
            avg_reward=0.0,
            execution_time=f"Tokens: {total_tokens:,}"
        )
        
        return GroupData(
            group_id="token_sequences",
            stats=stats,
            samples=samples
        )
    
    def _create_group_data_from_compressed(self, compressed_group: CompressedTokenGroup) -> GroupData:
        """Create GroupData from compressed token group"""
        # Convert stats to GroupStats format
        token_stats = compressed_group['stats']
        stats = GroupStats(
            total_results=token_stats['total_sequences'],
            success_rate=f"{token_stats['total_sequences']}/{token_stats['total_sequences']} (100.0%)",
            avg_reward=0.0,  # Not applicable for token data
            execution_time=f"Tokens: {token_stats['total_tokens']:,}"
        )
        
        # Create samples from sequence info
        samples = []
        for seq_info in compressed_group['sequences_info']:
            sample = SampleData(
                sample_id=seq_info['sequence_id'],
                name=seq_info['display_id'],
                is_success=True,  # Token sequences don't have success/failure
                weighted_reward=0.0,
                token_data={
                    'compressed_group': compressed_group,
                    'sequence_info': seq_info
                },
                metadata={
                    'total_tokens': seq_info['length'],
                    'avg_logprob': token_stats['avg_logprob'],
                    'avg_entropy': token_stats['avg_entropy'],
                    'avg_advantage': token_stats['avg_advantage'],
                    'format_version': 'compressed_v1'
                }
            )
            samples.append(sample)
            
        return GroupData(
            group_id=compressed_group['group_id'],
            stats=stats,
            samples=samples
        )
    
    def _convert_rollout_to_sample(self, result: SingleRolloutResult, index: int) -> SampleData:
        """Convert SingleRolloutResult to SampleData"""
        # Process conversation
        conversation = []
        if result.messages:
            for msg in result.messages:
                msg_data = {
                    "role": msg.role,
                    "content": msg.get_text(),
                    "timestamp": "2025-07-05 14:30:00"  # Mock timestamp
                }
                
                # Add tool-specific fields for tool messages
                if msg.role == "tool":
                    if hasattr(msg, 'tool_name') and msg.tool_name:
                        msg_data["tool_name"] = msg.tool_name
                    if hasattr(msg, 'tool_arguments') and msg.tool_arguments:
                        msg_data["tool_arguments"] = msg.tool_arguments
                    if hasattr(msg, 'structured_output') and msg.structured_output:
                        msg_data["structured_output"] = msg.structured_output
                
                # Add turn-level rewards for assistant messages
                if hasattr(msg, 'turn_reward_components') and msg.turn_reward_components:
                    msg_data["rewards"] = msg.turn_reward_components.copy()
                    if hasattr(msg, 'weighted_turn_reward') and msg.weighted_turn_reward is not None:
                        msg_data["rewards"]["Weighted Turn Reward"] = msg.weighted_turn_reward
                    if hasattr(msg, 'turn_level_advantage') and msg.turn_level_advantage is not None:
                        msg_data["rewards"]["Turn Advantage"] = msg.turn_level_advantage
                    if hasattr(msg, 'outcome_level_advantage') and msg.outcome_level_advantage is not None:
                        msg_data["rewards"]["Outcome Advantage"] = msg.outcome_level_advantage
                    if hasattr(msg, 'emt_grpo_advantage') and msg.emt_grpo_advantage is not None:
                        msg_data["rewards"]["EMT-GRPO Advantage"] = msg.emt_grpo_advantage
                        
                conversation.append(msg_data)
        
        # Get system prompt
        system_prompt = None
        if result.messages:
            for msg in result.messages:
                if msg.role == "system":
                    system_prompt = msg.get_text()
                    break
        
        # Process metadata
        metadata = {}
        if result.token_stats:
            total_tokens = result.token_stats.total_tokens or 0
            completion_tokens = result.token_stats.num_completion_tokens or 0
            input_tokens = total_tokens - completion_tokens if total_tokens else 0
            
            metadata.update({
                "inputTokens": input_tokens,
                "outputTokens": completion_tokens,
                "totalTokens": total_tokens,
            })
            
        if result.execution_time:
            metadata["executionTime"] = f"{result.execution_time:.1f}s"
            
        # Add end_reason if available
        if hasattr(result, 'metadata') and result.metadata and 'end_reason' in result.metadata:
            metadata["endReason"] = result.metadata["end_reason"]
            
        return SampleData(
            sample_id=result.id or f"rollout_{index}",
            name=f"Generation {index} ({'✓' if result.is_success else '✗'})",
            is_success=result.is_success or False,
            weighted_reward=result.weighted_reward or 0.0,
            conversation=conversation,
            rewards=result.reward_components or {},
            system_prompt=system_prompt,
            metadata=metadata
        )
    
    def _generate_navigation_data(self, groups: Dict[str, GroupData]) -> Dict[str, Any]:
        """Generate navigation data for the app"""
        return {
            "group_names": list(groups.keys()),
            "total_groups": len(groups),
            "total_samples": sum(len(group.samples) for group in groups.values())
        }