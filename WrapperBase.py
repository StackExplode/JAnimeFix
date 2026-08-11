
#Wrapper Interface
class WrapperBase:
	def __init__(self, globaljson, json):
		self.globalsetting = globaljson
		self.setting = json
	
		
	def GetSetting(self, key, defaultval = None):
		return self.setting.get(key, defaultval)
	
	def GetGlobalSetting(self, key, defaultval = None):
		return self.globalsetting.get(key, defaultval)

	def LoadModel(self):
		pass
	
	def Process(self, anydata):
		return anydata
	
class UplcalerWrapperBase(WrapperBase):
	def __init__(self, globaljson, json):
		super().__init__(globaljson, json)
	
	def GetSize(self,w,h):
		pass
	
class InterpolatorWrapperBase(WrapperBase):
	def __init__(self, globaljson, json):
		super().__init__(globaljson, json)
	
	def GetFPS(self,fps):
		pass

