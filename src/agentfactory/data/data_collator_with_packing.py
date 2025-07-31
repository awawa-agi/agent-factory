import torch

class DataCollatorWithPacking:
    """Data collator with sequence packing for efficient LLM and MLLM training."""
    
    def __init__(self, processor=None):
        self.processor = processor
    
    def __call__(self, instances):
        # Extract required input sequences
        input_ids = [instance["input_ids"] for instance in instances]
        seq_lens = [len(ids) for ids in input_ids]
        
        # Calculate packed attention metadata
        cumsum_seq_lens = torch.cumsum(torch.tensor([0] + seq_lens, dtype=torch.int32), dim=0)
        max_seq_len = torch.tensor([max(seq_lens)], dtype=torch.int32)
        
        # Pack input sequences
        packed_input_ids = torch.cat(input_ids, dim=0).unsqueeze(0)
        packed_position_ids = self._handle_position_ids(instances)
        
        cumsum_seq_lens = cumsum_seq_lens.to(torch.int32)
        max_seq_len = max_seq_len.to(torch.int32)
        
        # Build core batch
        batch = {
            "input_ids": packed_input_ids,
            "attention_mask": torch.ones_like(packed_input_ids, dtype=torch.int32),
            "position_ids": packed_position_ids,
            "cu_seq_lens_q": cumsum_seq_lens.clone(),
            "cu_seq_lens_k": cumsum_seq_lens.clone(),
            "max_length_k": max_seq_len.clone(),
            "max_length_q": max_seq_len.clone(),
        }
        
        # Add training fields if present
        self._add_training_fields(batch, instances)
        
        # Add vision fields if present
        self._add_vision_fields(batch, instances)
        
        # Add indices if all instances have them
        if all('idx' in inst for inst in instances):
            batch['indices'] = torch.tensor([inst['idx'] for inst in instances])
        
        return batch
    
    def _handle_position_ids(self, instances):
        """Handle position_ids with auto-generation and dimension detection."""
        if "position_ids" in instances[0] and instances[0]["position_ids"] is not None:
            position_ids = [instance["position_ids"] for instance in instances]
            
            if position_ids[0].dim() == 1:
                return torch.cat(position_ids, dim=0) # normal rope
            elif position_ids[0].dim() == 3:
                return torch.cat(position_ids, dim=2) # qwen2.5 vl mrope
            else:
                raise ValueError(f"Unsupported position_ids dimension: {position_ids[0].dim()}")
        else:
            # Auto-generate 1D position sequences
            position_sequences = [torch.arange(len(instance["input_ids"])) for instance in instances]
            return torch.cat(position_sequences, dim=0).unsqueeze(0)
    
    def _add_training_fields(self, batch, instances):
        """Add training-specific fields if present in all instances."""
        training_fields = {
            "advantages": lambda x: torch.cat(x, dim=0).unsqueeze(0),
            "assistant_masks": lambda x: torch.cat(x, dim=0).unsqueeze(0),
            "old_per_token_logps": lambda x: torch.cat(x, dim=0),
        }
        
        for field, pack_fn in training_fields.items():
            if field in instances[0] and instances[0][field] is not None:
                values = [inst[field] for inst in instances]
                batch[field] = pack_fn(values)
    
    def _add_vision_fields(self, batch, instances):
        """Add vision fields if present (supports mixed LLM/MLLM batches)."""
        # Handle pixel values
        pixel_values = [inst["pixel_values"] for inst in instances 
                       if "pixel_values" in inst and inst["pixel_values"] is not None]
        if pixel_values:
            batch["pixel_values"] = torch.cat(pixel_values, dim=0)
        
        # Handle image grid metadata
        grid_values = [inst["image_grid_thw"] for inst in instances 
                      if "image_grid_thw" in inst and inst["image_grid_thw"] is not None]
        if grid_values:
            batch["image_grid_thw"] = torch.cat(grid_values, dim=0)