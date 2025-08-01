"""
Token data adapter for the unified visualizer
Handles conversion of token sequences to standardized format
"""

from typing import List, Dict, Any, Optional, Tuple
import html


class TokenAdapter:
    """Adapter for converting token data to unified format"""
    
    def __init__(self, chunk_size: int = 500):
        # chunk_size parameter kept for compatibility but not used
        pass
    
    def adapt_token_sequences(self, sequences_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Adapt token sequence data to standardized format
        
        Args:
            sequences_data: List of token sequence dictionaries
            
        Returns:
            Normalized token sequence list
        """
        # Normalize sequences to consistent format
        normalized_sequences = []
        for i, seq_data in enumerate(sequences_data):
            if isinstance(seq_data, dict) and "tokens" in seq_data:
                normalized_sequences.append(seq_data)
            else:
                # Handle legacy formats
                normalized = self.adapt_legacy_format([seq_data])[0] if seq_data else {}
                normalized_sequences.append(normalized)
        
        return normalized_sequences
    
    @staticmethod
    def _adapt_single_sequence(seq_data: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Adapt a single token sequence"""
        # Extract basic data
        tokens = seq_data.get("tokens", [])
        logprobs = seq_data.get("logprobs", [])
        entropies = seq_data.get("entropies", [])
        advantages = seq_data.get("advantages")
        assistant_masks = seq_data.get("assistant_masks")
        
        # Calculate advantages if not provided
        if advantages is None and logprobs:
            avg_logprob = sum(logprobs) / len(logprobs) if logprobs else 0
            advantages = [lp - avg_logprob for lp in logprobs]
        
        # Default assistant masks if not provided
        if assistant_masks is None:
            assistant_masks = [1] * len(tokens)
        
        # Process tokens for display
        processed_tokens = []
        for j, token in enumerate(tokens):
            display_token, newline_count = TokenAdapter._process_token_with_newlines(token)
            
            token_data = {
                "token": html.escape(display_token),
                "logprob": logprobs[j] if j < len(logprobs) else 0.0,
                "entropy": entropies[j] if j < len(entropies) else 0.0,
                "advantage": advantages[j] if advantages and j < len(advantages) else 0.0,
                "assistant_mask": assistant_masks[j] if j < len(assistant_masks) else 1,
                "newline_count": newline_count,
                "index": j
            }
            processed_tokens.append(token_data)
        
        # Calculate metadata
        metadata = TokenAdapter._calculate_token_metadata(processed_tokens)
        
        return {
            "sequence_id": seq_data.get("sequence_id", f"seq_{index}"),
            "display_id": seq_data.get("display_id", seq_data.get("label", f"Sequence {index + 1}")),
            "tokens": processed_tokens,
            "metadata": metadata,
            "collapse_groups": TokenAdapter._identify_collapse_groups(assistant_masks),
            "original_data": seq_data  # Keep reference to original data
        }
    
    @staticmethod
    def _process_token_with_newlines(token: str) -> Tuple[str, int]:
        """Process tokens containing newlines for display"""
        original_token = token
        newline_count = 0
        
        while token.endswith('\n'):
            newline_count += 1
            token = token[:-1]
        
        if original_token.strip() == '' and '\n' in original_token:
            display_token = '\\n' * original_token.count('\n')
        else:
            display_token = token.replace('\n', '\\n')
            if newline_count > 0:
                display_token += '\\n' * newline_count
        
        return display_token, newline_count
    
    @staticmethod
    def _identify_collapse_groups(assistant_masks: List[int], min_length: int = 3) -> List[Optional[int]]:
        """Identify consecutive groups of non-assistant tokens that should be collapsed"""
        collapse_groups = [None] * len(assistant_masks)
        group_id = 0
        i = 0
        
        while i < len(assistant_masks):
            if assistant_masks[i] == 0:
                start = i
                while i < len(assistant_masks) and assistant_masks[i] == 0:
                    i += 1
                
                if i - start >= min_length:
                    for j in range(start, i):
                        collapse_groups[j] = group_id
                    group_id += 1
            else:
                i += 1
        
        return collapse_groups
    
    @staticmethod
    def _calculate_token_metadata(tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metadata for a token sequence"""
        if not tokens:
            return {
                "total_tokens": 0,
                "assistant_tokens": 0,
                "avg_logprob": 0.0,
                "avg_entropy": 0.0,
                "avg_advantage": 0.0
            }
        
        assistant_tokens = [t for t in tokens if t["assistant_mask"] == 1]
        
        metadata = {
            "total_tokens": len(tokens),
            "assistant_tokens": len(assistant_tokens),
        }
        
        if assistant_tokens:
            metadata.update({
                "avg_logprob": sum(t["logprob"] for t in assistant_tokens) / len(assistant_tokens),
                "avg_entropy": sum(t["entropy"] for t in assistant_tokens) / len(assistant_tokens),
                "avg_advantage": sum(t["advantage"] for t in assistant_tokens) / len(assistant_tokens),
            })
        else:
            metadata.update({
                "avg_logprob": 0.0,
                "avg_entropy": 0.0,
                "avg_advantage": 0.0,
            })
        
        return metadata
    
    def adapt_legacy_format(self, legacy_data) -> List[Dict[str, Any]]:
        """
        Adapt legacy token data formats to the new unified format
        
        Args:
            legacy_data: Legacy format data (various possible formats)
        """
        # Handle different legacy formats
        if isinstance(legacy_data, dict):
            # Single sequence as dict
            return [TokenAdapter._adapt_legacy_dict(legacy_data)]
        
        elif isinstance(legacy_data, list):
            if len(legacy_data) == 0:
                return []
            
            # Check if it's list of dicts (new format) or list of lists (old format)
            if isinstance(legacy_data[0], dict):
                # Already in dict format, just adapt each one
                return [TokenAdapter._adapt_legacy_dict(seq) for seq in legacy_data]
            
            elif isinstance(legacy_data[0], list):
                # Old format: [tokens, logprobs, entropies, advantages, assistant_masks]
                return [TokenAdapter._adapt_legacy_list(legacy_data)]
            
            else:
                # Assume it's a single sequence as [tokens, logprobs, ...]
                return [TokenAdapter._adapt_legacy_list(legacy_data)]
        
        return []
    
    @staticmethod
    def _adapt_legacy_dict(seq_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt a legacy dictionary format"""
        # If it already looks like our format, pass through
        if "tokens" in seq_dict and isinstance(seq_dict["tokens"], list):
            return seq_dict
        
        # Otherwise, try to extract what we can
        return {
            "tokens": seq_dict.get("tokens", []),
            "logprobs": seq_dict.get("logprobs", []),
            "entropies": seq_dict.get("entropies", []),
            "advantages": seq_dict.get("advantages"),
            "assistant_masks": seq_dict.get("assistant_masks"),
            "display_id": seq_dict.get("display_id", seq_dict.get("label", "Sequence")),
            "sequence_id": seq_dict.get("sequence_id", "seq_0")
        }
    
    @staticmethod
    def _adapt_legacy_list(seq_list: List[Any]) -> Dict[str, Any]:
        """Adapt a legacy list format [tokens, logprobs, entropies, ...]"""
        seq_dict = {
            "tokens": seq_list[0] if len(seq_list) > 0 else [],
            "logprobs": seq_list[1] if len(seq_list) > 1 else [],
            "entropies": seq_list[2] if len(seq_list) > 2 else [],
            "display_id": "Sequence",
            "sequence_id": "seq_0"
        }
        
        if len(seq_list) > 3:
            seq_dict["advantages"] = seq_list[3]
        
        if len(seq_list) > 4:
            seq_dict["assistant_masks"] = seq_list[4]
        
        return seq_dict