import sys, json

try:
  config: dict = json.load(open('./config.json'))
except:
  sys.exit('config.json not found')