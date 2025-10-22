from typing import Dict, List, Optional, Tuple, Union, Iterable
import torch
import numpy as np
from PIL import Image

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

def resize(
        image: Image,
        size: Tuple[int, int],
        resample: Image.Resampling = None,
        reducing_gap: Optional[int]= None
) -> np.ndarray:
    height, width = size
    resized_image = image.resize((width, height), resample=resample, reducing_gap=reducing_gap)
    return np.array(resized_image)


def rescale(
        image: np.ndarray,
        scale: float,
        dtype: np.dtype = np.float32
) -> np.ndarray:
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image

def normalize(
        image: np.ndarray,
        mean: Union[float, Iterable[float]],
        std: Union[float, Iterable[float]]
) -> np.ndarray:
    mean = np.array(mean , dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image

def process_image(
        image: List[Image.Image],
        size: Dict[str, int] = None,
        resample : Image.Resampling = None,
        rescale_factor: float = None,
        image_mean : Optional[Union[float, List[float]]] = None,
        image_std : Optional[Union[float, List[float]]] = None,
)-> List[np.ndarray]:
    height, width = size[0], size[1]
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]

    images = [np.array(image) for image in images]

    images = [rescale(image=image, scale=rescale_factor) for image in images]

    images = [normalize(image, mean=image_mean, std=image_std) for image in images]

    images = [image.transpose(2,0,1) for image in images] #HWC to CHW

    return images



class PaliGemmaProcessor:

    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size:int):
        super().__init__()

        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        #Tokenizer 
        tokens_to_add = { "additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)

        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ] #These tokens are used in object detection tasks

        EXTRA_TOKENS = [
            f"<loc{i:03d}>" for i in range(128)
        ] #These tokens are used in object segmentation tasks

        tokenizer.add_special_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        #We will add the BOS and EOS tokens to the image sequence
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer
