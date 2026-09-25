#!/usr/bin/env python3
""" This module generates a textual description (caption)
of a given image using a pre-trained BLIP Vision-Language Model.
"""
import transformers
import PIL


def image_captioner(model, image_path, max_new_tokens):
    """
    Generate a textual description (caption) of a given image
    using a pre-trained BLIP Vision-Language Model.

    Args:
        model: Name of the pre-trained image captioning model to use.
        image_path: Path to the image file to caption.
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        caption: Generated textual description of the image.
    """
    # Load the BLIP processor and pre-trained model.
    processor = transformers.BlipProcessor.from_pretrained(model)
    model = transformers.BlipConditionalGeneration.from_pretrained(
        model)

    # Convert the image into PyTorch tensors using the processor.
    image = PIL.Image.open(image_path).convert("RGB")

    # Generate caption tokens from the model with the processed inputs.
    inputs = processor(image=image,
                       return_tensors="pt")

    # Decode the generated tokens into readable text, skipping special tokens.
    output = model.generate(**inputs,
                            max_new_tokens=max_new_tokens)

    caption = processor.decode(output[0],
                               skip_special_tokens=True)

    return caption
