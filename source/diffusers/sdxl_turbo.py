from .model import Model
from diffusers import AutoPipelineForText2Image
from ..config import config

class SDXLTurbo(Model):
  def __init__(self):
    super().__init__(config['sdxl-turbo']['pretrained-name'], AutoPipelineForText2Image)

  def generate(self, prompt: str, kwargs: dict={}):
    if self.diffuser == None: return None
    try:
      images = self.diffuser(prompt=prompt, **kwargs).images
      return images
    except Exception as e:
      print(e)
      return None