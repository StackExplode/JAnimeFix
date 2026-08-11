import json

#Wrapper Interface
class WrapperBase:
	def __init__(self, globaljson, json):
		self.globalsetting = globaljson
		self.setting = json
	
	def CorrectSetting(self):
		pass
		
	def GetSetting(self, key, defaultval = None):
		return self.setting.get(key, defaultval)
	
	def GetGlobalSetting(self, key, defaultval = None):
		return self.globalsetting.get(key, defaultval)

	def LoadModel(self):
		pass
	
	def Process(self, anydata):
		return anydata