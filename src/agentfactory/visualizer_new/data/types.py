"""
Type definitions for the token visualizer system
"""

from typing import Dict, List, Any, Optional, Union, TypedDict


# Token data type definitions
class TokenSequenceInput(TypedDict):
    """Input format for a single token sequence"""
    tokens: List[str]
    logprobs: List[float]
    entropies: List[float]
    advantages: Optional[List[float]]
    assistant_masks: Optional[List[int]]
    sequence_id: Optional[str]
    display_id: Optional[str]


class QuantizedSequenceData(TypedDict):
    """Quantized token sequence data for compression"""
    tokens: List[str]
    logprobs_q: List[int]      # int16 quantized (*1000)
    entropies_q: List[int]     # int16 quantized (*1000)  
    advantages_q: List[int]    # int16 quantized (*1000)
    assistant_masks: List[bool]
    sequence_id: str
    display_id: str
    length: int


class TokenGroupStats(TypedDict):
    """Statistics for a token group (pre-calculated in Python)"""
    total_sequences: int
    total_tokens: int
    avg_logprob: float       # Pre-calculated, no need to decompress
    avg_entropy: float       # Pre-calculated, no need to decompress
    avg_advantage: float     # Pre-calculated, no need to decompress
    logprob_range: List[float]  # [min, max]
    entropy_range: List[float]  # [min, max]
    advantage_range: List[float]  # [min, max]


class TokenSequenceInfo(TypedDict):
    """Lightweight sequence info (no compressed data)"""
    sequence_id: str
    display_id: str
    length: int


class CompressedTokenGroup(TypedDict):
    """Compressed token group data"""
    group_id: str
    stats: TokenGroupStats
    sequences_info: List[TokenSequenceInfo]  # Lightweight info for UI
    compressed_data: str                     # base64(gzip(quantized_data))


# Input format definitions
TokenDataInput = Union[
    List[TokenSequenceInput],                    # Single group (auto-wrapped)
    Dict[str, List[TokenSequenceInput]]         # Multiple groups
]