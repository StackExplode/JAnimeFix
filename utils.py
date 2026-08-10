import importlib

class Utils:
	@staticmethod
	def CreateInstance(path, *args, **kwargs):
		module_name, class_name = path.rsplit(".", 1)
		
		module = importlib.import_module(module_name)
		cls = getattr(module, class_name)
		
		return cls(*args, **kwargs)