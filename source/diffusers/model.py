import torch

class Model:
  def __init__(self, pretrainName: str, diffuser):
    self.pretrainName = pretrainName
    self.diffuser = diffuser

  def load(self):
    self.diffuser = self.diffuser.from_pretrained(
      self.pretrainName,
      torch_dtype=torch.float16,
      variant='fp16'
    )
    self.diffuser = self.diffuser.to("cuda")

  def export(self):
    return self