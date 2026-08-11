import cv2
import numpy as np
import torch
import queue
import subprocess
import threading
from tqdm import tqdm
from typing_extensions import deprecated
import shlex

class Worker:
	def __init__(self,config,upscaler,interpolator):
		self.config = config
		self.upscaler = upscaler
		self.interpolator = interpolator
		self.output_resize = config.get("output_resize", "none")
		self.output_short_edge = config.get("output_short_edge", 1080)
		self.output_fps = config.get("output_fps", 30)
		self.ffmpeg_encoder = config.get("ffmpeg_encoder", "h264_nvenc")
		self.ffmpeg_preset = config.get("ffmpeg_preset", "slow")
		self.ffmpeg_crf = config.get("ffmpeg_crf", "18")
		self.ffmpeg_pixfmt = config.get("ffmpeg_pixfmt", "yuv420p")
		self.ffmpeg_extra = config.get("ffmpeg_extra", "")
		self.device = config.get("device", "cuda:0")
		self.frame_window_size = config.get("frame_window_size", 12)
		self.temp_dir = config.get("temp_dir", "temp")
		
	
	# --- 线程 1：读取线程 ---
	def video_reader_thread(self, cap, read_queue):
		while True:
			ret, frame = cap.read()
			if not ret:
				read_queue.put(None)  # 放入结束信号
				break
			read_queue.put(frame)
	
	# --- 线程 3：写入线程 ---
	def video_writer_thread(self, process, write_queue):
		while True:
			frame_bytes = write_queue.get()
			if frame_bytes is None:
				break
			process.stdin.write(frame_bytes)
	
	# --- 显卡到内存的转换 (放到最后一步) ---
	def tensors_to_bytes(self, tensor_list):
		"""把 GPU 里的高清张量转成可以送给 FFmpeg 的字节流"""
		out_bytes = []
		for tensor in tensor_list:
			# GPU -> CPU -> NumPy
			tensor = tensor.squeeze(0).float().cpu().clamp_(0, 1)
			img = tensor.numpy()
			img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))
			img = (img * 255.0).round().astype(np.uint8)
			out_bytes.append(img.tobytes())
			
		return out_bytes
	
	def _get_ffmpeg_out_param(self, stage): #stage: 1=upscale, 2=interpolate, 3=special
		if self.output_resize != "none":
			resizestr = f"scale='if(gt(iw,ih),-2,{self.output_short_edge})':'if(gt(iw,ih),{self.output_short_edge},-2)':flags={self.output_resize}"
		else:
			resizestr = ""
		if self.output_resize != 0:
			fpsstr = f"fps={self.output_fps}"
		else:
			fpsstr = ""

		if stage == 1:
			return f"-vf \"{resizestr}\"" if resizestr != "" else ""
		elif stage == 2:
			return f"-vf \"{fpsstr}\"" if fpsstr != "" else ""
		elif stage == 3:
			if resizestr != "" and fpsstr != "":
				return f"-vf \"{resizestr},{fpsstr}\""
			elif resizestr != "":
				return f"-vf \"{resizestr}\""
			elif fpsstr != "":
				return f"-vf \"{fpsstr}\""
			else:
				return ""
			
		
	
	def _get_ffmpeg_param(self,stage,isfinal,output_path,w,h,fps):
		if isfinal:
			ffmpeg_cmd = [
				'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
				'-s', f'{w}x{h}', '-pix_fmt', 'bgr24', '-r', str(fps),
				'-i', '-',
				'-c:v', self.ffmpeg_encoder,
				'-preset', self.ffmpeg_preset,
				'-cq' if self.ffmpeg_encoder.endswith("nvenc") else "-crf", str(self.ffmpeg_crf),
				'-pix_fmt', self.ffmpeg_pixfmt,
			]
			
			if self.device.startswith("cuda"):
				dev_num = int(self.device.split(":")[1]) if ":" in self.device else 0
				ffmpeg_cmd.extend(['-gpu', str(dev_num)])
			
			ext=self._get_ffmpeg_out_param(stage)
			if ext != "":
				ffmpeg_cmd.extend(shlex.split(ext))
			
			if len(self.ffmpeg_extra) > 0:
				ffmpeg_cmd.extend(shlex.split(self.ffmpeg_extra))
			
			ffmpeg_cmd.append(output_path)
		else:
			ffmpeg_cmd = [
				'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
				'-s', f'{w}x{h}', '-pix_fmt', 'bgr24', '-r', str(fps),
				'-i', '-',
				'-c:v', "hevc_nvenc",
				'-preset', "p6",
				'-spatial-aq', '1',
				'-rc', 'vbr_hq',
				'-cq' , "17",
				'-pix_fmt', "yuv420p10le",
				'-spatial-aq', '1',
				'-temporal-aq', '1',
				'-rc-lookahead', '32',
			]
			
			if self.device.startswith("cuda"):
				dev_num = int(self.device.split(":")[1]) if ":" in self.device else 0
				ffmpeg_cmd.extend(['-gpu', str(dev_num)])
			
			ext = self._get_ffmpeg_out_param(stage)
			if ext != "":
				ffmpeg_cmd.extend(shlex.split(ext))
			
			ffmpeg_cmd.append(output_path)
		return ffmpeg_cmd
	
	@deprecated("废弃！")
	def process_video(self, input_path, output_path):
		pass
			
	def ProcessUpscale(self, input_path, output_path, isfinal):
		upscaler = self.upscaler
		upscaler.LoadModel()
		
		cap = cv2.VideoCapture(input_path)
		org_fps = cap.get(cv2.CAP_PROP_FPS)
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		orig_w, orig_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		

		up_w, up_h = upscaler.GetSize(orig_w, orig_h)
		

		ffmpeg_cmd = self._get_ffmpeg_param(stage=1,isfinal=isfinal, output_path=output_path, w=up_w, h=up_h, fps=org_fps)
		
		process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
		
		# 建立多线程队列
		read_queue = queue.Queue(maxsize=30)
		write_queue = queue.Queue(maxsize=30)
		
		# 启动工作线程
		threading.Thread(target=self.video_reader_thread, args=(cap, read_queue), daemon=True).start()
		writer_t = threading.Thread(target=self.video_writer_thread, args=(process, write_queue), daemon=True)
		writer_t.start()
		
		window_buffer = []
		pbar = tqdm(total=total_frames, desc="放大进度", unit="帧")
		try:
			# 主线程专心做 GPU 推断
			while True:
				frame = read_queue.get()
				if frame is None:  # 读完了
					break
				
				window_buffer.append(frame)
				pbar.update(1)
				
				if len(window_buffer) == self.frame_window_size:
					# 1. AnimeSR 放大 (返回 GPU Tensors)
					sr_tensors = upscaler.Process(window_buffer)
					
					# 3. 截断最后的重复帧
					# 原本窗口内要输出的帧数为：len(window_buffer) - 1
					# 插帧后，这部分对应的实际帧数需要乘以插帧倍率
					frames_to_write = len(window_buffer) - 1
					#frames_to_write_after_interp = frames_to_write * self.interpolation_factor
					
					# 4. 截断后下放到 CPU 并送入写入队列
					final_bytes = self.tensors_to_bytes(sr_tensors[:frames_to_write])
					for b in final_bytes:
						write_queue.put(b)
					
					window_buffer = [window_buffer[-1]]
			
			# 处理视频末尾剩余的收尾帧
			if len(window_buffer) > 1:
				sr_tensors = upscaler.Process(window_buffer)

				final_bytes = self.tensors_to_bytes(sr_tensors, )
				for b in final_bytes:
					write_queue.put(b)
			
			elif len(window_buffer) == 1:
				# 只剩孤立的一帧，只能放大，无法进行“两两之间”的插帧
				sr_tensors = upscaler.Process(window_buffer)
				final_bytes = self.tensors_to_bytes([sr_tensors[0]])
				write_queue.put(final_bytes[0])
		
		except KeyboardInterrupt:
			print("\n[警告] 检测到手动中断 (Ctrl+C)！正在安全保存已处理的视频片段，请勿再次强制退出...")
		
		finally:
			write_queue.put(None)  # 通知写入线程结束
			writer_t.join()  # 等待最后的视频编码完成
			cap.release()
			pbar.close()
			
			print("正在等待 FFmpeg 封装视频...")
			process.stdin.close()
			process.wait()
			tempstr = "" if isfinal else "临时目录"
			print(f"放大完成！视频已保存至{tempstr}:", output_path)
	
	def ProcessInterpolate(self, input_path, output_path):
		raise NotImplementedError("尚未实现")
	
	def MergeOtherTracks(self, input_path, output_path):
		pass