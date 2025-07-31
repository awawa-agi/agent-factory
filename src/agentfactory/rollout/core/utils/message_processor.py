import os
import uuid
from pathlib import Path
from ..messages import Messages, BaseMessage
from ..content import TextContent, ImageContent

def generate_random_filename(
    save_dir: str,
) -> Path:
    rand_filename = uuid.uuid4().hex[:8]
    local_save_path = Path(save_dir, f"{rand_filename}.jpg")
    while os.path.exists(local_save_path):
        rand_filename = uuid.uuid4().hex[:8]
        local_save_path = Path(save_dir, f"{rand_filename}.jpg")
    return local_save_path

def process_image_in_message(
    message: BaseMessage,
    do_resize: bool = True,
    max_pixels: int = 28 * 28 * 128,
    min_pixels: int = 28 * 28 * 4,
    add_image_tag: bool = True,
    do_save: bool = False,
    local_save_dir: str | None = None,
    display_save_dir: str | None = None,
) -> BaseMessage:
    
    new_contents = []
    for content in message.content:
        if content.type != "image_url":
            new_contents.append(content)
            continue

        if do_save:
            if local_save_dir is None:
                raise ValueError("local_save_dir is required when do_save is True")
            if display_save_dir is None:
                raise ValueError("display_save_dir is required when do_save is True")

            rand_filename = generate_random_filename(local_save_dir)
            save_path = Path(local_save_dir, rand_filename)
            display_path = Path(display_save_dir, rand_filename)
            content.save_to_file(save_path)

        if do_resize:
            try:
                content = content.resize_smart(max_pixels=max_pixels, min_pixels=min_pixels)
            except Exception as e:
                content = TextContent(text="[Image cannot be displayed: size constraints not met or aspect ratio too extreme]")

        if add_image_tag and content.type == "image_url":
            tag_text = (
                f"<image "
                f"{f'path={display_path} ' if do_save else ''}"
                f"display_size={content.width}x{content.height}>"
            )
            new_contents.append(TextContent(text=tag_text))
            
        new_contents.append(content)

    return message.model_copy(update={"content": new_contents})

def process_image_in_messages(
    messages: Messages,
    do_resize: bool = True,
    max_pixels: int = 28 * 28 * 128,
    min_pixels: int = 28 * 28 * 4,
    add_image_tag: bool = True,
    do_save: bool = False,
    local_save_dir: str | None = None,
    display_save_dir: str | None = None,
) -> Messages:
    
    new_messages = []

    for msg in messages:
        new_msg = process_image_in_message(
            msg,
            do_resize=do_resize,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            add_image_tag=add_image_tag,
            do_save=do_save,
            local_save_dir=local_save_dir,
            display_save_dir=display_save_dir,
        )
        new_messages.append(new_msg)

    return Messages(new_messages)

def _truncate_text(text: str, max_length: int, front_portion: float = 0.5) -> str:
    """
    Truncate text keeping portions from front and back
    
    Args:
        text: Text to truncate
        max_length: Maximum allowed length
        front_portion: Ratio of text to keep from front (0.1-0.9)
        
    Returns:
        Truncated text with ellipsis separator
    """
    # Return original text if within length limit
    if len(text) <= max_length:
        return text
    
    # Clamp front_portion to valid range
    front_portion = max(0.1, min(0.9, front_portion))
    
    # Calculate lengths for front and back portions
    ellipsis_marker = "\n\n[...content truncated...]\n\n"
    ellipsis_length = len(ellipsis_marker)
    
    front_length = int(max_length * front_portion)
    back_length = max_length - front_length - ellipsis_length
    
    # Handle edge case where back portion would be too small
    if back_length <= 0:
        simple_ellipsis = "[...]"
        return text[:max_length - len(simple_ellipsis)] + simple_ellipsis
    
    # Extract front and back portions
    front_part = text[:front_length]
    back_part = text[-back_length:]
    
    return f"{front_part}{ellipsis_marker}{back_part}"

def truncate_overlong_text_in_message(
    message: BaseMessage,
    max_length: int = 1000,
) -> BaseMessage:
    new_contents = []
    for content in message.content:
        if content.type != "text":
            new_contents.append(content)
            continue
        content.text = _truncate_text(content.text, max_length)
        new_contents.append(content)
    return message.model_copy(update={"content": new_contents})

def truncate_overlong_text_in_messages(
    messages: Messages,
    max_length: int = 1000,
) -> Messages:
    new_messages = []
    for msg in messages:
        new_msg = truncate_overlong_text_in_message(msg, max_length)
        new_messages.append(new_msg)
    return Messages.model_validate(new_messages)