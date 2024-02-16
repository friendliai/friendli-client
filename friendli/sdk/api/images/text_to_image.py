# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Text to Image API."""

# pylint: disable=line-too-long, no-name-in-module

from __future__ import annotations

import os
from typing import Optional, Type

from friendli.schema.api.v1.codegen.text_to_image_pb2 import V1TextToImageRequest
from friendli.schema.api.v1.images.image import Image, ImageResponseFormatParam
from friendli.sdk.api.base import AsyncServingAPI, ServingAPI
from friendli.utils.compat import model_parse


class TextToImage(ServingAPI[Type[V1TextToImageRequest]]):
    """Text to image API."""

    @property
    def _api_path(self) -> str:
        return "v1/text-to-image"

    @property
    def _method(self) -> str:
        return "POST"

    @property
    def _content_type(self) -> str:
        boundary = os.urandom(16).hex().encode("ascii")
        return f"multipart/form-data; boundary={boundary.decode('ascii')}"

    @property
    def _request_pb_cls(self) -> Type[V1TextToImageRequest]:
        return V1TextToImageRequest

    def create(
        self,
        *,
        prompt: str,
        model: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_outputs: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        response_format: Optional[ImageResponseFormatParam] = None,
    ) -> Image:
        """Creates an image from the text prompt.

        Args:
            prompt (str): Text prompt describing the image to generate.
            model (Optional[str], optional): Code of the model to use.
            negative_prompt (Optional[str], optional): A text specifying what you don't want in your image(s). Defaults to None.
            num_outputs (Optional[int], optional): The number of images to generate. Must be between 1 and 16. Only 1 output will be generated when not provided. Defaults to None.
            num_inference_steps (Optional[int], optional): The number of inference steps for denoising process. 50 steps will be taken when not provided. Defaults to None.
            guidance_scale (Optional[float], optional): Guidance scale to control how much generation process adheres to the text prompt. When not provided, it is set to 7.5. Defaults to None.
            seed (Optional[int], optional): Seed to control random procedure. If nothing is given, the API generate the seed randomly. Defaults to None.
            response_format (Optional[ImageResponseFormatParam], optional): The format in which the generated images are returned. Defaults to None.

        Returns:
            Image: Data of the generated image.

        """
        request_dict = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_outputs": num_outputs,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "response_format": response_format,
        }
        response = self._request(data=request_dict, stream=False, model=model)

        return model_parse(Image, response.json())


class AsyncTextToImage(AsyncServingAPI[Type[V1TextToImageRequest]]):
    """Text to image API."""

    @property
    def _api_path(self) -> str:
        return "v1/text-to-image"

    @property
    def _method(self) -> str:
        return "POST"

    @property
    def _content_type(self) -> str:
        boundary = os.urandom(16).hex().encode("ascii")
        return f"multipart/form-data; boundary={boundary.decode('ascii')}"

    @property
    def _request_pb_cls(self) -> Type[V1TextToImageRequest]:
        return V1TextToImageRequest

    async def create(
        self,
        *,
        prompt: str,
        model: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_outputs: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        response_format: Optional[ImageResponseFormatParam] = None,
    ) -> Image:
        """Creates an image from the text prompt asychronously.

        Args:
            prompt (str): Text prompt describing the image to generate.
            model (Optional[str], optional): Code of the model to use.
            negative_prompt (Optional[str], optional): A text specifying what you don't want in your image(s). Defaults to None.
            num_outputs (Optional[int], optional): The number of images to generate. Must be between 1 and 16. Only 1 output will be generated when not provided. Defaults to None.
            num_inference_steps (Optional[int], optional): The number of inference steps for denoising process. 50 steps will be taken when not provided. Defaults to None.
            guidance_scale (Optional[float], optional): Guidance scale to control how much generation process adheres to the text prompt. When not provided, it is set to 7.5. Defaults to None.
            seed (Optional[int], optional): Seed to control random procedure. If nothing is given, the API generate the seed randomly. Defaults to None.
            response_format (Optional[ImageResponseFormatParam], optional): The format in which the generated images are returned. Defaults to None.

        Returns:
            Image: Data of the generated image.

        """
        request_dict = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_outputs": num_outputs,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "response_format": response_format,
        }
        response = await self._request(data=request_dict, stream=False, model=model)

        return model_parse(Image, response.json())
