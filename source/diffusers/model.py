import torch

class Model:
  def __init__(self, pretrainName: str, diffuser):
    self.pretrainName = pretrainName
    self.diffuser = diffuser
    self.loaded = False
    self.opts = {}

  def load(self, opts: dict={}):
    try:
      self.diffuser = self.diffuser.from_pretrained(
        self.pretrainName,
        torch_dtype=torch.float16,
        variant='fp16',
        **opts
      )
      self.diffuser = self.diffuser.to("cuda")
      self.loaded = True
    except Exception as e:
      print(e)