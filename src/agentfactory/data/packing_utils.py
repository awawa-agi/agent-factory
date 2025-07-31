"""
Utilities for handling packed sequence data in AgentFactory.

This module provides functions to unpack sequences that have been packed together
for efficient training, particularly useful for handling variable-length sequences
in RL training scenarios.
"""

import torch
from typing import List, Dict, Union, Optional, Any


def unpack_sequences(packed_out: torch.Tensor, cu_seq_lens: torch.Tensor) -> List[torch.Tensor]:
    """
    Unpack sequences from packed tensor using cumulative sequence lengths.
    
    Args:
        packed_out: (B=1, L_total, D) or (L_total, D) - packed forward output
        cu_seq_lens: (N+1,) - cumulative sequence lengths
    
    Returns:
        list[Tensor] - N tensors, each with shape (L_i, D)
    """
    # Remove dummy batch dimension if present
    if packed_out.dim() == 3 and packed_out.size(0) == 1:
        packed_out = packed_out.squeeze(0)
    
    # Compute individual sequence lengths (vectorized)
    lengths = torch.diff(cu_seq_lens)
    
    # Split tensor efficiently
    return list(torch.split(packed_out, lengths.tolist(), dim=0))


def unpack_batch(
    batch_data: Dict[str, torch.Tensor], 
    cu_seq_lens: torch.Tensor, 
    indices: Optional[Union[torch.Tensor, List[int]]] = None, 
    return_dict: bool = True
) -> Union[List[Dict[str, torch.Tensor]], Dict[int, Dict[str, torch.Tensor]]]:
    """
    Unpack all sequences in a batch with optional sequence IDs.
    
    Args:
        batch_data: dict containing packed tensors
        cu_seq_lens: (N+1,) cumulative sequence lengths  
        indices: (N,) sequence IDs, optional
        return_dict: bool, if True and indices provided, return dict mapping seq_id -> data
                     if False, always return list
    
    Returns:
        list[dict] - N dicts, each containing unpacked data for one sequence
        OR dict[int, dict] - if indices provided and return_dict=True, maps seq_id -> unpacked data
    """
    # Fields to unpack (1D sequences)
    sequence_fields = [
        'input_ids', 'logps', 'entropies', 
        'assistant_masks', 'advantages'
    ]
    
    # Unpack all sequence fields
    unpacked = {}
    for field in sequence_fields:
        if field in batch_data:
            unpacked[field] = unpack_sequences(batch_data[field], cu_seq_lens)
    
    # Organize by sequence
    num_sequences = len(cu_seq_lens) - 1
    
    if indices is not None and return_dict:
        # Return dict mapping seq_id -> data
        result = {}
        for i in range(num_sequences):
            seq_id = indices[i].item() if torch.is_tensor(indices) else indices[i]
            seq_data = {field: tensors[i] for field, tensors in unpacked.items()}
            result[seq_id] = seq_data
        return result
    else:
        # Return list of dicts
        result = []
        for i in range(num_sequences):
            seq_data = {field: tensors[i] for field, tensors in unpacked.items()}
            # Add sequence ID to data if indices provided
            if indices is not None:
                seq_id = indices[i].item() if torch.is_tensor(indices) else indices[i]
                seq_data['seq_id'] = seq_id
            result.append(seq_data)
        return result