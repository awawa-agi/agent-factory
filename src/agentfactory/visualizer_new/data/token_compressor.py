"""
Token data quantization and compression utilities
Handles efficient storage and transmission of token visualization data
"""

import json
import gzip
import base64
from typing import List, Dict, Any
import numpy as np

from .types import (
    TokenSequenceInput, 
    QuantizedSequenceData, 
    TokenGroupStats,
    TokenSequenceInfo,
    CompressedTokenGroup
)


class TokenDataCompressor:
    """Handles quantization and compression of token data"""
    
    @staticmethod
    def quantize_values(values: List[float]) -> List[int]:
        """Convert float values to int16 quantization (*1000)"""
        # Convert to numpy for efficient operations
        arr = np.array(values, dtype=np.float32)
        # Quantize to int16 range with *1000 precision
        quantized = (arr * 1000).astype(np.int16)
        return quantized.tolist()
    
    @staticmethod
    def dequantize_values(values: List[int]) -> List[float]:
        """Convert int16 quantized values back to float (/1000)"""
        arr = np.array(values, dtype=np.int16)
        dequantized = arr.astype(np.float32) / 1000.0
        return dequantized.tolist()
    
    @staticmethod
    def quantize_sequence(sequence: TokenSequenceInput) -> QuantizedSequenceData:
        """Quantize a single token sequence"""
        # Handle missing advantages
        advantages = sequence.get('advantages') or [0.0] * len(sequence['tokens'])
        
        # Handle missing assistant_masks  
        assistant_masks = sequence.get('assistant_masks') or [1] * len(sequence['tokens'])
        
        # Ensure all arrays have same length
        length = len(sequence['tokens'])
        logprobs = sequence['logprobs'][:length]
        entropies = sequence['entropies'][:length] 
        advantages = advantages[:length]
        assistant_masks = assistant_masks[:length]
        
        return QuantizedSequenceData(
            tokens=sequence['tokens'],
            logprobs_q=TokenDataCompressor.quantize_values(logprobs),
            entropies_q=TokenDataCompressor.quantize_values(entropies),
            advantages_q=TokenDataCompressor.quantize_values(advantages),
            assistant_masks=[bool(mask) for mask in assistant_masks],
            sequence_id=sequence.get('sequence_id', f'seq_{id(sequence)}'),
            display_id=sequence.get('display_id', 'Sequence'),
            length=length
        )
    
    @staticmethod
    def calculate_group_stats(sequences: List[TokenSequenceInput]) -> TokenGroupStats:
        """Pre-calculate statistics for a group of sequences"""
        if not sequences:
            return TokenGroupStats(
                total_sequences=0,
                total_tokens=0,
                avg_logprob=0.0,
                avg_entropy=0.0, 
                avg_advantage=0.0,
                logprob_range=[0.0, 0.0],
                entropy_range=[0.0, 0.0],
                advantage_range=[0.0, 0.0]
            )
        
        # Collect all values for statistics
        all_logprobs = []
        all_entropies = []
        all_advantages = []
        total_tokens = 0
        
        for seq in sequences:
            # Handle missing data
            advantages = seq.get('advantages') or [0.0] * len(seq['tokens'])
            assistant_masks = seq.get('assistant_masks') or [1] * len(seq['tokens'])
            
            # Only count assistant tokens for stats
            for i, (logprob, entropy, advantage, mask) in enumerate(zip(
                seq['logprobs'], seq['entropies'], advantages, assistant_masks
            )):
                if mask:  # Only assistant tokens
                    all_logprobs.append(logprob)
                    all_entropies.append(entropy)
                    all_advantages.append(advantage)
            
            total_tokens += len(seq['tokens'])
        
        # Calculate statistics
        if all_logprobs:
            avg_logprob = float(np.mean(all_logprobs))
            avg_entropy = float(np.mean(all_entropies))
            avg_advantage = float(np.mean(all_advantages))
            
            logprob_range = [float(np.min(all_logprobs)), float(np.max(all_logprobs))]
            entropy_range = [float(np.min(all_entropies)), float(np.max(all_entropies))]
            advantage_range = [float(np.min(all_advantages)), float(np.max(all_advantages))]
        else:
            avg_logprob = avg_entropy = avg_advantage = 0.0
            logprob_range = entropy_range = advantage_range = [0.0, 0.0]
        
        return TokenGroupStats(
            total_sequences=len(sequences),
            total_tokens=total_tokens,
            avg_logprob=round(avg_logprob, 4),
            avg_entropy=round(avg_entropy, 4),
            avg_advantage=round(avg_advantage, 4),
            logprob_range=[round(x, 4) for x in logprob_range],
            entropy_range=[round(x, 4) for x in entropy_range],
            advantage_range=[round(x, 4) for x in advantage_range]
        )
    
    @staticmethod
    def compress_group(group_id: str, sequences: List[TokenSequenceInput]) -> CompressedTokenGroup:
        """Compress a group of token sequences"""
        # Pre-calculate statistics
        stats = TokenDataCompressor.calculate_group_stats(sequences)
        
        # Create lightweight sequence info
        sequences_info = []
        quantized_sequences = []
        
        for seq in sequences:
            # Quantize sequence data
            quantized = TokenDataCompressor.quantize_sequence(seq)
            quantized_sequences.append(quantized)
            
            # Extract lightweight info
            sequences_info.append(TokenSequenceInfo(
                sequence_id=quantized['sequence_id'],
                display_id=quantized['display_id'],
                length=quantized['length']
            ))
        
        # Compress quantized data
        json_data = json.dumps(quantized_sequences, separators=(',', ':'))
        original_size = len(json_data.encode('utf-8'))
        compressed_bytes = gzip.compress(json_data.encode('utf-8'))
        compressed_base64 = base64.b64encode(compressed_bytes).decode('ascii')
        
        # Calculate and print compression statistics
        compressed_size = len(compressed_bytes)
        base64_size = len(compressed_base64)
        compression_ratio = compressed_size / original_size if original_size > 0 else 0.0
        space_saved = (1 - compression_ratio) * 100
        
        print(f"🗜️  Compression Stats for group '{group_id}':")
        print(f"   Original size:    {original_size:,} bytes")
        print(f"   Compressed size:  {compressed_size:,} bytes")
        print(f"   Base64 size:      {base64_size:,} bytes")
        print(f"   Compression ratio: {compression_ratio:.3f}")
        print(f"   Space saved:      {space_saved:.1f}%")
        print(f"   Total sequences:  {len(sequences)}")
        print(f"   Total tokens:     {sum(len(seq.get('tokens', [])) for seq in sequences):,}")
        print("")
        
        return CompressedTokenGroup(
            group_id=group_id,
            stats=stats,
            sequences_info=sequences_info,
            compressed_data=compressed_base64
        )
    
    @staticmethod
    def decompress_group(compressed_group: CompressedTokenGroup) -> List[QuantizedSequenceData]:
        """Decompress a group of token sequences (for testing/verification)"""
        compressed_bytes = base64.b64decode(compressed_group['compressed_data'])
        decompressed_data = gzip.decompress(compressed_bytes).decode('utf-8')
        quantized_sequences = json.loads(decompressed_data)
        return quantized_sequences
    
    @staticmethod
    def estimate_compression_ratio(sequences: List[TokenSequenceInput]) -> Dict[str, Any]:
        """Estimate compression efficiency"""
        if not sequences:
            return {"original_size": 0, "compressed_size": 0, "ratio": 0.0}
        
        # Estimate original size (JSON)
        original_json = json.dumps(sequences)
        original_size = len(original_json.encode('utf-8'))
        
        # Create compressed version
        compressed = TokenDataCompressor.compress_group("test", sequences)
        compressed_size = len(compressed['compressed_data'])
        
        ratio = compressed_size / original_size if original_size > 0 else 0.0
        
        return {
            "original_size": original_size,
            "compressed_size": compressed_size, 
            "compression_ratio": round(ratio, 3),
            "space_saved": round((1 - ratio) * 100, 1)
        }