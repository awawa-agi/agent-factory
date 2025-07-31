from copy import deepcopy
from typing import Optional, Union, Unpack, Tuple
from PIL import Image

import torch
from transformers import Qwen2_5_VLProcessor
from transformers.processing_utils import AllKwargsForChatTemplate
from .qwen_vl_utils import extract_vision_info, fetch_image, fetch_video, get_rope_index_25



class Qwen2_5_VLProcessorWithAssistantMask(Qwen2_5_VLProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_vision_info(
        self,
        conversations: list[dict] | list[list[dict]],
        return_video_kwargs: bool = False,
    ) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:
        """
        Extract vision info from conversation and add vision num_token to each content.
        """
        vision_infos = extract_vision_info(conversations)
        ## Read images or videos
        image_inputs = []
        video_inputs = []
        video_sample_fps_list = []
        for vision_info in vision_infos:
            if "image" in vision_info or "image_url" in vision_info:
                fetched_image = fetch_image(vision_info)    
                image_inputs.append(fetched_image)
                vision_info["num_token"] = self._get_num_multimodal_tokens(
                    [[fetched_image.width, fetched_image.height]],
                    min_pixels=self.image_processor.min_pixels,
                    max_pixels=self.image_processor.max_pixels,
                    patch_size=self.image_processor.patch_size,
                    merge_size=self.image_processor.merge_size,
                ).num_image_tokens[0]
            elif "video" in vision_info:
                video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
                video_sample_fps_list.append(video_sample_fps)
                video_inputs.append(video_input)
            else:
                raise ValueError("image, image_url or video should in content.")
        if len(image_inputs) == 0:
            image_inputs = None
        if len(video_inputs) == 0:
            video_inputs = None
        if return_video_kwargs:
            return image_inputs, video_inputs, {'fps': video_sample_fps_list}
        return image_inputs, video_inputs


    def apply_chat_template_with_assistant_mask(
        self,
        conversation: Union[list[dict[str, str]], list[list[dict[str, str]]]],
        chat_template: Optional[str] = None,
        return_position_ids: bool = False,
    ) -> dict:

        conversation = deepcopy(conversation)

        images, videos = self.process_vision_info(conversation)
        
        image_inputs = {}
        image_grid_thw = None
        if images is not None:
            image_inputs = self.image_processor(images=images)
            image_grid_thw = image_inputs["image_grid_thw"]

        if videos is not None:
            raise NotImplementedError("Video is not supported yet.")

        text_inputs = self.tokenizer.apply_chat_template(
            conversation, tokenize=True, return_dict=True, return_assistant_tokens_mask=True
        )

        out_dict = {
            **text_inputs,
            **image_inputs,
        }

        if return_position_ids:
            position_ids, _ = get_rope_index_25(
                input_ids=torch.tensor([text_inputs['input_ids']]),
                image_grid_thw=image_grid_thw,
            )
            out_dict['position_ids'] = position_ids

        return out_dict