from .model import Model
from diffusers import StableDiffusionXLPipeline
from ..config import config

class SDXLXL(Model):
  def __init__(self):
    super().__init__(config['sdxl-xl']['pretrained-name'], StableDiffusionXLPipeline)

  def generate(self, prompt: str, kwargs: dict={}):
    if self.diffuser == None: return None
    try:
      images = self.diffuser(prompt=prompt, **kwargs).images
      return images
    except Exception as e:
      print(e)
      return None