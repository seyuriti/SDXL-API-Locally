from .model import Model
from diffusers import AutoPipelineForText2Image
from ..config import config

class SDXLTurbo(Model):
  def __init__(self):
    super().__init__(config['sdxl-turbo']['pretrained-name'], AutoPipelineForText2Image)

  def generate(self, prompt: str, opts: dict={}):
    try:
      if self.loaded == False:
        raise Exception("Model not loaded")
      if self.diffuser == None: return None
      images = self.diffuser(prompt=prompt, **opts).images
      self.opts = opts
      return images
    except Exception as e:
      print(e)
      return []