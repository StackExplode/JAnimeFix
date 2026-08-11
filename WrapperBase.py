
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
	
	def get_warmup_frames(self) -> int:
		"""
		[抽象接口] 获取模型在切片并行时的“热身”帧数。
		- 纯 CNN 模型 (如 RealESRGAN, RIFE)：返回 0
		- RNN/时序模型 (如 AnimeSR)：返回 20 (或你的窗口大小)
		"""
		return 0
	
	def Process(self, anydata):
		return anydata
	
class UpcalerWrapperBase(WrapperBase):
	def __init__(self, globaljson, json):
		super().__init__(globaljson, json)
	
	def GetSize(self,w,h):
		pass
	
class InterpolatorWrapperBase(WrapperBase):
	def __init__(self, globaljson, json):
		super().__init__(globaljson, json)
	
	def GetFPS(self,fps):
		pass

