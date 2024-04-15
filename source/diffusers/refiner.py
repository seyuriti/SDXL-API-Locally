import torch
from diffusers import DiffusionPipeline
from ..config import config
from .model import Model

class Refiner(Model):
  def __init__(self, base: Model):
    super().__init__(config['refiner']['pretrained-name'], DiffusionPipeline)
    self.base = base

  def load(self, opts: dict={}):
    super().load(opts.update({
      'text_encoder_2': self.base.diffuser.text_encoder_2,
      'vae': self.base.diffuser.vae
    }))

  def refine(self, images: list, prompt: str):
    if self.diffuser is None:
      return None
    images = self.base.diffuser(
      prompt=prompt,
      image=images,
      **self.opts
    ).images
    return images