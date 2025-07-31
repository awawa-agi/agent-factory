import torch
from torch.utils.data import Dataset
from typing import Any, Optional, TypedDict, NotRequired, Union
from transformers import PreTrainedTokenizer
from transformers.processing_utils import ProcessorMixin
from loguru import logger


class DataItem(TypedDict):
    messages: list[dict[str, Any]]
    num_loss_tokens: int
    num_tokens: int
    advantage: NotRequired[float]
    turn_advantages: NotRequired[list[float]]


def transform_masks(assistant_masks):
    """Transform assistant masks to group indices.
    
    Example: [0,0,1,1,0,1,1,1] -> [0,0,1,1,0,2,2,2]
    """
    result = []
    group = 0
    prev = 0
    
    for mask in assistant_masks:
        if mask == 1 and prev == 0:
            group += 1
        result.append(mask * group)
        prev = mask
    
    return result


class RLDataset(Dataset):
    def __init__(
        self, 
        data: list[DataItem], 
        processing_class: PreTrainedTokenizer | ProcessorMixin, 
        use_turn_advantage: bool = False
    ):
        self.data = data
        self.processing_class = processing_class
        self.use_turn_advantage = use_turn_advantage
        
    def __len__(self):
        return len(self.data)
    
    def get_sequence_lengths(self):
        return [d['num_tokens'] for d in self.data]
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Apply chat template
        encoded = self._apply_chat_template(sample['messages'])
        
        # Convert to tensors
        input_ids = torch.tensor(encoded['input_ids'])
        attention_mask = torch.tensor(encoded['attention_mask'], dtype=torch.int32)
        assistant_masks = torch.tensor(encoded['assistant_masks'], dtype=torch.int32)
        
        # Calculate advantages
        advantages = self._calculate_advantages(
            input_ids, assistant_masks, sample
        )
        
        # Build output item
        item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'assistant_masks': assistant_masks,
            'advantages': advantages,
            'idx': idx
        }
        
        # Add optional fields
        for key in ['pixel_values', 'image_grid_thw', 'position_ids']:
            if key in encoded:
                item[key] = encoded[key]
                
        return item
    
    def _apply_chat_template(self, messages):
        """Apply chat template with proper method selection."""
        if hasattr(self.processing_class, 'apply_chat_template_with_assistant_mask'):
            return self.processing_class.apply_chat_template_with_assistant_mask(
                messages,
                return_position_ids=True
            )
        else:
            return self.processing_class.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
            )
    
    def _calculate_advantages(self, input_ids, assistant_masks, sample):
        """Calculate advantages based on use_turn_advantage flag."""
        if not self.use_turn_advantage:
            # Simple advantage: same value for all positions
            return torch.ones_like(input_ids, dtype=torch.float32) * sample['advantage']
        
        # Turn-specific advantages
        if 'turn_advantages' not in sample:
            raise ValueError(
                "turn_advantages is required when use_turn_advantage is True"
            )
        
        transformed_masks = torch.tensor(transform_masks(assistant_masks.tolist()))
        num_groups = int(transformed_masks.max().item())
        
        # Validate turn_advantages length
        if num_groups != len(sample['turn_advantages']):
            raise ValueError(
                f"Expected {num_groups} turn_advantages, "
                f"got {len(sample['turn_advantages'])}"
            )
        
        # Initialize advantages with zeros
        advantages = torch.zeros_like(input_ids, dtype=torch.float32)
        
        # Assign advantages for each group (skip group 0 which are non-assistant tokens)
        for group_idx in range(num_groups):
            mask = transformed_masks == (group_idx + 1)
            advantages[mask] = sample['turn_advantages'][group_idx]
        
        return advantages