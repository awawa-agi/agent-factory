import torch
from trl.trainer.utils import pad

class DataCollatorWithPaddingAndVision:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_ids = pad([f['input_ids'] for f in features], padding_value=self.processor.tokenizer.pad_token_id)
        assistant_masks = pad([f['assistant_masks'] for f in features], padding_value=0)
        attention_mask = pad([f['attention_mask'] for f in features], padding_value=0)

        pixel_values = [f['pixel_values'] for f in features if f.get('pixel_values', None) is not None]
        pixel_values = torch.cat(pixel_values) if pixel_values else None
        image_grid_thw = [f['image_grid_thw'] for f in features if f.get('image_grid_thw', None) is not None]
        image_grid_thw = torch.cat(image_grid_thw) if image_grid_thw else None
        
        position_ids = None
        if 'position_ids' in features[0]:
            position_ids = pad([f['position_ids'] for f in features], padding_value=1)
        
        advantages = torch.tensor([f['advantage'] for f in features])

        collated_features = {
            'input_ids': input_ids,
            'assistant_masks': assistant_masks,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw,
            'position_ids': position_ids,
            'advantages': advantages,
        }

        if 'idx' in features[0]:
            collated_features['indices'] = torch.tensor([f['idx'] for f in features])

        if 'old_per_token_logps' in features[0]:
            old_per_token_logps = pad([f['old_per_token_logps'] for f in features], padding_value=0)
            collated_features['old_per_token_logps'] = old_per_token_logps

        return collated_features