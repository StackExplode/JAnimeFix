import importlib
import os
import sys

class Utils:
	
	_tempfiles = []
	
	
	@staticmethod
	def CreateInstance(path, *args, **kwargs):
		module_name, class_name = path.rsplit(".", 1)
		
		module = importlib.import_module(module_name)
		cls = getattr(module, class_name)
		
		return cls(*args, **kwargs)
	
	@staticmethod
	def InitTempDir(config):
		if not os.path.exists(config.get("temp_dir")):
			os.makedirs(config.get("temp_dir"))

	@staticmethod
	def AddTempFile(path):
		Utils._tempfiles.append(path)
		
	@staticmethod
	def AddTempFileEx(root,path):
		full_path = os.path.join(root, path)
		Utils._tempfiles.append(full_path)
		
	@staticmethod
	def AddTempFiles(paths):
		Utils._tempfiles.extend(paths)
		
	@staticmethod
	def AddTempFilesEx(root, paths):
		full_paths = [os.path.join(root, p) for p in paths]
		Utils._tempfiles.extend(full_paths)
		
	@staticmethod
	def CleanupTempFiles():
		for path in Utils._tempfiles:
			if os.path.exists(path):
				try:
					os.remove(path)
				except Exception as e:
					print(f"无法删除临时文件 {path}: {e}")
	
		self.temp_files.clear()  # 遍历完之后一次性清空
	
	