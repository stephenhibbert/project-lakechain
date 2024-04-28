#  Copyright (C) 2023 Amazon.com, Inc. or its affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
import copy
import sys

from io import BytesIO
from urllib.parse import urlparse
import torch
import numpy as np
from PIL import Image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, HeunDiscreteScheduler, AutoencoderKL
from compel import Compel, ReturnedEmbeddingsType
import boto3

s3_client = boto3.client('s3')

# The device used for inference.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Environment variables.
CACHE_DIR = os.environ.get('CACHE_DIR')
TARGET_BUCKET = os.getenv('PROCESSED_FILES_BUCKET')

depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas", cache_dir=CACHE_DIR).to("cuda")
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas", cache_dir=CACHE_DIR)

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
).to("cuda")

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", 
    torch_dtype=torch.float16,
    cache_dir=CACHE_DIR
).to("cuda")

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
    cache_dir=CACHE_DIR
).to("cuda")
# print(pipe.scheduler.compatibles)
pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)

compel = Compel(
  tokenizer=[pipe.tokenizer, pipe.tokenizer_2] ,
  text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
  returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
  requires_pooled=[False, True]
)

def get_depth_map(image, feature_extractor, depth_estimator):
    rgb_image = image.convert("RGB")
    image = feature_extractor(images=rgb_image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image

def image_to_bytes(image):
  """
  Converts an image to a PNG byte array.
  :param image: The image to convert.
  :param mime_type: The MIME type to convert the image to.
  :return: The image as a byte array and its MIME type.
  """
  with BytesIO() as bytes_io:
    image.save(bytes_io, 'PNG')
    bytes_io.seek(0)
    return bytes_io.getvalue(), 'image/png'

def get_image(image: Image, prompt: str = "", steps: int = 10) -> str:
  """
  :param image: the image to generate a description for.
  :param prompt: the text prompt for the image generation
  :param steps: the number of denioising steps
  :return: the encoded images
  """
  conditioning, pooled = compel(prompt)
  depth_image = get_depth_map(image, feature_extractor, depth_estimator)
  generator = torch.manual_seed(33)
  
  # https://huggingface.co/docs/diffusers/en/api/pipelines/controlnet#diffusers.StableDiffusionControlNetImg2ImgPipeline.__call__
  generated_images = pipe(
      prompt_embeds=conditioning,
      pooled_prompt_embeds=pooled,
      negative_prompt='monochrome, lowres, bad anatomy, worst quality, low quality, fantasy, clutter, physically implausible placements, unrealistic geometry, objects that defy gravity, smudgy, blurry',
      image=depth_image,
      strength=0.8,
      num_inference_steps=steps,
      controlnet_conditioning_scale=0.4,
      generator=generator,
  ).images

  return generated_images

def get_metadata_from_image(event: dict, image: Image) -> dict:
  """
  Computes a description for the given image and stores it
  in the description field of the document metadata.
  :event: the CloudEvent associated with the image.
  :image: the image loaded in memory.
  :returns: a dictionary containing the extracted metadata.
  """
  s3_client  = boto3.client('s3')
  document   = event['data']['document']
  chain_id   = event['data']['chainId']
  output_key = f"{chain_id}/generated.png"

  # Computing a description for the image.
  generated_images = get_image(image)

  # Convert the image to a byte array.
  data, mime_type = image_to_bytes(generated_images[0])

  upload_result = s3_client.put_object(
    Bucket=TARGET_BUCKET,
    Key=output_key,
    Body=data,
    ContentType=mime_type
  )
  
  # Set the new document in the event.
  event['data']['document'] = {
    'url': f"s3://{TARGET_BUCKET}/{output_key}",
    'type': mime_type,
    'size': sys.getsizeof(data),
    'etag': upload_result['ETag'].replace('"', '')
  }
  
  return event
