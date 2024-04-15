from .model import Model
from diffusers import StableDiffusionXLPipeline
from ..config import config

class SDXLXL(Model):
  def __init__(self):
    super().__init__(config['sdxl-xl']['pretrained-name'], StableDiffusionXLPipeline)

  def generate(self, prompt: str, opts: dict={}):
    if self.diffuser == None: return None
    try:
      images = self.diffuser(prompt=prompt, **opts).images
      self.opts = opts
      return images
    except Exception as e:
      print(e)
      return []