import cv2
import numpy as np
import torch
import queue
import subprocess
import threading
from tqdm import tqdm

class Worker:
	def __init__(self,config,upscaler,interpolator):
		self.config = config
		self.upscaler = upscaler
		self.interpolator = interpolator
		self.upscale_factor = config.get("upscale_factor", 4)
		self.interpolation_factor = config.get("interpolation_factor", 2)
		self.output_short_edge = config.get("output_short_edge", 1080)
		self.ffmpeg_encoder = config.get("ffmpeg_encoder", "h264_nvenc")
		self.ffmpeg_preset = config.get("ffmpeg_preset", "slow")
		self.ffmpeg_crf = config.get("ffmpeg_crf", "18")
		self.ffmpeg_pixfmt = config.get("ffmpeg_pixfmt", "yuv420p")
		self.ffmpeg_extra = config.get("ffmpeg_extra", "")
		self.device = config.get("device", "cuda:0")
		self.frame_window_size = config.get("frame_window_size", 12)
		
	
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
	def tensors_to_bytes(self, tensor_list, short_edge):
		"""把 GPU 里的高清张量转成可以送给 FFmpeg 的字节流"""
		out_bytes = []
		for tensor in tensor_list:
			# GPU -> CPU -> NumPy
			tensor = tensor.squeeze(0).float().cpu().clamp_(0, 1)
			img = tensor.numpy()
			img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))
			img = (img * 255.0).round().astype(np.uint8)
			
			# 缩放
			h, w = img.shape[:2]
			if h < w:
				new_h, new_w = short_edge, int(w * (short_edge / h))
			else:
				new_w, new_h = short_edge, int(h * (short_edge / w))
			
			img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
			out_bytes.append(img.tobytes())
		return out_bytes
	
	def process_video(self, input_path, output_path):
		upscaler, interpolator = (self.upscaler, self.interpolator)
		
		upscaler.LoadModel()
		interpolator.LoadModel()
		
		cap = cv2.VideoCapture(input_path)
		orig_fps = cap.get(cv2.CAP_PROP_FPS)
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		orig_w, orig_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		out_fps = orig_fps * self.interpolation_factor
		
		temp_h, temp_w = orig_h * self.upscale_factor, orig_w * self.upscale_factor
		out_h = self.output_short_edge if temp_h < temp_w else int(temp_h * (self.output_short_edge / temp_w))
		out_w = int(temp_w * (self.output_short_edge / temp_h)) if temp_h < temp_w else self.output_short_edge
		out_w, out_h = out_w if out_w % 2 == 0 else out_w + 1, out_h if out_h % 2 == 0 else out_h + 1
		
		ffmpeg_cmd = [
			'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
			'-s', f'{out_w}x{out_h}', '-pix_fmt', 'bgr24', '-r', str(out_fps),
			'-i', '-',
			'-c:v', self.ffmpeg_encoder,
			'-preset', self.ffmpeg_preset,
			'-cq' if self.ffmpeg_encoder.endswith("nvenc") else "-crf", self.ffmpeg_crf,
			'-pix_fmt', self.ffmpeg_pixfmt,
		]
		
		if self.device.startswith("cuda"):
			dev_num = int(self.device.split(":")[1]) if ":" in self.device else 0
			ffmpeg_cmd.extend(['-gpu', str(dev_num)])
		
		if len(self.ffmpeg_extra) > 0:
			ffmpeg_cmd.append(self.ffmpeg_extra)
		
		ffmpeg_cmd.append(output_path)
		
		process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
		
		# 建立多线程队列
		read_queue = queue.Queue(maxsize=30)
		write_queue = queue.Queue(maxsize=30)
		
		# 启动工作线程
		threading.Thread(target=self.video_reader_thread, args=(cap, read_queue), daemon=True).start()
		writer_t = threading.Thread(target=self.video_writer_thread, args=(process, write_queue), daemon=True)
		writer_t.start()
		
		window_buffer = []
		pbar = tqdm(total=total_frames, desc="处理进度", unit="帧")
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
					
					# 2. RIFE 占位插帧 (真实增加帧数，对齐输出 fps)
					interp_tensors = interpolator.Process(sr_tensors)
					
					# 3. 截断最后的重复帧
					# 原本窗口内要输出的帧数为：len(window_buffer) - 1
					# 插帧后，这部分对应的实际帧数需要乘以插帧倍率
					frames_to_write = len(window_buffer) - 1
					frames_to_write_after_interp = frames_to_write * self.interpolation_factor
					
					# 4. 截断后下放到 CPU 并送入写入队列
					final_bytes = self.tensors_to_bytes(interp_tensors[:frames_to_write_after_interp], self.output_short_edge)
					for b in final_bytes:
						write_queue.put(b)
					
					window_buffer = [window_buffer[-1]]
			
			# 处理视频末尾剩余的收尾帧
			if len(window_buffer) > 1:
				sr_tensors = upscaler.Process(window_buffer)
				# 同样需要经过占位插帧扩充数量
				interp_tensors = interpolator.Process(sr_tensors)
				final_bytes = self.tensors_to_bytes(interp_tensors, self.output_short_edge)
				for b in final_bytes:
					write_queue.put(b)
			
			elif len(window_buffer) == 1:
				# 只剩孤立的一帧，只能放大，无法进行“两两之间”的插帧
				sr_tensors = upscaler.Process(window_buffer)
				final_bytes = self.tensors_to_bytes([sr_tensors[0]], self.output_short_edge)
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
			print("处理完成！视频已保存至:", output_path)