import json

#Wrapper Interface
class WrapperBase:
	def __init__(self, globaljson, json):
		self.globalsetting = json.loads(globaljson)
		self.setting = json.loads(json)
		
	def GetSetting(self, key, defaultval = None):
		return self.setting.get(key, defaultval)
	
	def GetGlobalSetting(self, key, defaultval = None):
		return self.globalsetting.get(key, defaultval)

	def LoadModel(self):
		raise NotImplementedError("LoadModel method not implemented.")
	
	def Process(self, frames):
		raise NotImplementedError("Process method not implemented.")