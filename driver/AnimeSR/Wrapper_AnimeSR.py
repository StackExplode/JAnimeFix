import os
import sys

import torchvision.transforms.functional as TF
sys.modules['torchvision.transforms.functional_tensor'] = TF


import cv2
import torch
import numpy as np
from WrapperBase import *

# 导入官方提供的工具函数
from utils.inference_base import get_base_argument_parser, get_inference_model

class Wrapper_AnimeSR(UplcalerWrapperBase):
	def __init__(self, globaljson, json):
		super().__init__(globaljson, json)
		self.device = torch.device(self.GetGlobalSetting("device", "cuda:0"))
		self.netscale = self.GetSetting("netscale", 4)
		self.ishalf = self.GetSetting("ishalf", True)
		self.mod_scale = self.GetSetting("mod_scale", 4)
		self.modelpath = self.GetSetting("model_path", "models/AnimeSR_v2.pth")
		self.isprecompile = self.GetSetting("isprecompile", True)
		self.model = None
	
	def GetSize(self,w,h):
		return w * self.netscale, h * self.netscale
	
	def LoadModel(self):
		"""
				使用官方提供的 get_inference_model 来加载模型。
				"""
		print("正在加载 AnimeSR 模型...")
		# 1. 获取官方的默认参数解析器
		parser = get_base_argument_parser()
		
		# 2. parse_known_args([]) 传入空列表，只获取默认参数配置，避免读取到外面主程序的 sys.argv
		args, _ = parser.parse_known_args([])
		
		# 3. 覆盖官方解析器中的关键参数，指向我们自定义的路径和倍率
		args.model_path = self.modelpath
		args.netscale = self.netscale
		
		# 4. 调用官方逻辑初始化模型并挂载到指定设备
		model = get_inference_model(args, self.device)
		if self.ishalf:
			model = model.half()
		print("正在对模型核心计算图进行 JIT 编译 (这需要一些时间)...")
		if hasattr(torch, 'compile') and self.isprecompile:
			model.cell = torch.compile(model.cell)
		
		self.model = model
	
	def _img_to_tensor(self, img):
		"""
		将 OpenCV 的 NumPy 图像转换为模型所需的 Tensor。
		为了不依赖外部的 basicsr 库，这里直接用纯 NumPy+PyTorch 实现官方的预处理逻辑。
		"""
		# 归一化
		img = img.astype(np.float32) / 255.
		
		# Mod Crop: 确保宽高可以被 4 整除
		h, w = img.shape[:2]
		h_pad, w_pad = h % self.mod_scale, w % self.mod_scale
		if h_pad != 0 or w_pad != 0:
			img = img[:h - h_pad, :w - w_pad, :]
		
		# BGR 转 RGB，并转换通道顺序 (HWC -> CHW)
		img = img[:, :, [2, 1, 0]]
		tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
		
		# 增加 Batch 维度并搬运到显卡
		return tensor.unsqueeze(0).to(self.device).half() if self.ishalf else tensor.unsqueeze(0).to(self.device)
	
	def _tensor_to_img(self, tensor):
		"""
		将显存中的 Tensor 还原为 OpenCV 格式的 NumPy 图像。
		"""
		tensor = tensor.squeeze(0).float().cpu().clamp_(0, 1)
		img = tensor.numpy()
		
		# CHW 转 HWC，并从 RGB 转回 BGR
		img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))
		img = (img * 255.0).round().astype(np.uint8)
		return img
	
	@torch.no_grad()
	def Process(self, frames):
		frames_bgr = frames
		"""
		[修改版] 处理连续帧序列，直接返回保留在 GPU 显存中的 Tensor 列表。
		彻底消除 PCIe 内存搬运损耗。
		"""
		num_imgs = len(frames_bgr)
		if num_imgs == 0:
			return []
		
		out_tensors = []
		
		# 1. 准备首帧
		prev = self._img_to_tensor(frames_bgr[0])
		cur = prev
		nxt_idx = min(1, num_imgs - 1)
		nxt = self._img_to_tensor(frames_bgr[nxt_idx])
		
		c, h, w = prev.size()[-3:]
		state = prev.new_zeros(1, 64, h, w)
		out_state = prev.new_zeros(1, c, h * self.netscale, w * self.netscale)
		
		# 2. 循环推进推理
		for idx in range(num_imgs):
			cat_tensor = torch.cat((prev, cur, nxt), dim=1)
			out_state, state = self.model.cell(cat_tensor, out_state, state)
			
			# 【关键修改】不再调用 self._tensor_to_img，直接克隆并保留在显存中
			out_tensors.append(out_state.clone())
			
			prev = cur
			cur = nxt
			next_read_idx = min(idx + 2, num_imgs - 1)
			nxt = self._img_to_tensor(frames_bgr[next_read_idx])
		
		return out_tensors  # 返回的是 GPU Tensors 列表