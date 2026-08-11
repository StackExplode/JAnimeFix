import os

import cv2
import numpy as np
import torch
import queue
import subprocess
import threading
from tqdm import tqdm
from typing_extensions import deprecated
import shlex

from utils import Utils


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
		if self.output_fps != 0:
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
			
	def ProcessUpscale_old(self, input_path, output_path, isfinal):
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
		pbar = tqdm(total=total_frames,smoothing=0.1, desc="放大进度", unit="帧")
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
	
	def _progress_listener(self, q, total_frames, desc):
		"""独立的进度条监听线程，解决多进程下控制台乱码闪烁的问题"""
		pbar = tqdm(total=total_frames, desc=desc, unit="帧", smoothing=0.1)
		for _ in iter(q.get, None):
			pbar.update(1)
		pbar.close()
	
	def _run_in_chunks(self, stage, input_path, output_path, isfinal, wrapper, out_w, out_h, out_fps, base_name,
	                   desc="处理进度"):
		"""
		统一的多进程切片调度核心。无论是放大还是插帧都复用此通道。
		"""
		chunk_num = self.config.get("chunk_num_upscale", 1)
		print(f"视频将以 {chunk_num} 个切片并行处理，请关注你的显存是否足够...")
		
		cap = cv2.VideoCapture(input_path)
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		cap.release()
		
		import math
		import multiprocessing as mp
		from utils import Utils
		
		frames_per_chunk = math.ceil(total_frames / chunk_num)
		chunk_args = []
		temp_chunk_files = []
		
		manager = mp.Manager()
		progress_queue = manager.Queue()
		
		# 1. 任务切割与临时文件注册
		for i in range(chunk_num):
			start_frame = i * frames_per_chunk
			end_frame = min((i + 1) * frames_per_chunk, total_frames)
			if start_frame >= total_frames:
				break
			
			# 以原文件名为基础派生安全切片名
			chunk_out_path = os.path.join(self.temp_dir, f"{base_name}_{desc}_chunk_{i}.mp4")
			temp_chunk_files.append(chunk_out_path)
			
			# 【生命周期管理】立刻注册零件，交由主程序的 Cleanup 统一销毁
			Utils.AddTempFile(chunk_out_path)
			
			ffmpeg_cmd = self._get_ffmpeg_param(stage, isfinal, chunk_out_path, out_w, out_h, out_fps)
			
			chunk_args.append((
				input_path, start_frame, end_frame,
				wrapper, ffmpeg_cmd, progress_queue,
				self.frame_window_size
			))
		
		# 2. 拉起监听器与子进程 (强制使用 spawn 隔离 CUDA)
		listener_t = threading.Thread(target=self._progress_listener, args=(progress_queue, total_frames, desc))
		listener_t.start()
		
		ctx = mp.get_context('spawn')
		with ctx.Pool(processes=len(chunk_args)) as pool:
			pool.starmap(self._chunk_processor, chunk_args)
		
		progress_queue.put(None)
		listener_t.join()
		
		# 3. 极速无损合并 (Concat Demuxer)
		print(f"\n[{desc}] 所有切片处理完毕，正在无损合并视频...")
		concat_list_path = os.path.join(self.temp_dir, f"{base_name}_{desc}_concat_list.txt")
		Utils.AddTempFile(concat_list_path)  # 同样注册 txt
		
		with open(concat_list_path, "w", encoding="utf-8") as f:
			for chunk_file in temp_chunk_files:
				abs_path = os.path.abspath(chunk_file).replace("\\", "/")
				f.write(f"file '{abs_path}'\n")
		
		concat_cmd = [
			'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
			'-i', concat_list_path, '-c', 'copy', output_path
		]
		subprocess.run(concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		
		# 完美贯彻“自动挡清理”，无需任何 os.remove 硬编码
		return output_path
	
	@staticmethod
	def tensors_to_bytes_static(tensor_list):
		"""必须升级为静态方法，供隔离的子进程调用"""
		out_bytes = []
		for tensor in tensor_list:
			tensor = tensor.squeeze(0).float().cpu().clamp_(0, 1)
			img = tensor.numpy()
			img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))
			img = (img * 255.0).round().astype(np.uint8)
			out_bytes.append(img.tobytes())
		return out_bytes
	
	@staticmethod
	def _chunk_processor(input_path, start_frame, end_frame, wrapper, ffmpeg_cmd, progress_queue, window_size):
		"""
		子进程内部处理逻辑。包含绝对坐标映射的热身帧精准丢弃算法。
		"""
		# 在子进程中实例化模型避免显存冲突
		wrapper.LoadModel()
		
		# 若模型基类中未定义 get_warmup_frames，则默认回退 0 帧
		warmup_frames = wrapper.get_warmup_frames() if hasattr(wrapper, 'get_warmup_frames') else 0
		actual_start_read = max(0, start_frame - warmup_frames)
		
		cap = cv2.VideoCapture(input_path)
		cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start_read)
		
		process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
		
		window_buffer = []
		current_read_idx = actual_start_read
		
		while current_read_idx < end_frame:
			ret, frame = cap.read()
			if not ret:
				break
			
			window_buffer.append(frame)
			current_read_idx += 1  # 累加，此时 current_read_idx 代表“下一帧将要读取的索引”
			
			if len(window_buffer) == window_size:
				sr_tensors = wrapper.Process(window_buffer)
				frames_to_write = len(window_buffer) - 1
				
				# 降维处理，转化为 Byte 数组
				final_bytes = Worker.tensors_to_bytes_static(sr_tensors[:frames_to_write])
				
				# 【算法核心】利用当前索引，逆推出当前数组内每一张画面的绝对帧坐标
				for k, b in enumerate(final_bytes):
					abs_frame_idx = current_read_idx - len(window_buffer) + k
					# 如果该画面属于“有效区”（不属于向左探取的热身区），则真正写入
					if abs_frame_idx >= start_frame:
						process.stdin.write(b)
						progress_queue.put(1)
				
				# 滑动窗口
				window_buffer = [window_buffer[-1]]
		
		# 收尾处理 (处理那些由于到达 end_frame 而未填满 window_size 的尾巴)
		if len(window_buffer) > 1:
			sr_tensors = wrapper.Process(window_buffer)
			final_bytes = Worker.tensors_to_bytes_static(sr_tensors)
			for k, b in enumerate(final_bytes):
				abs_frame_idx = current_read_idx - len(window_buffer) + k
				if abs_frame_idx >= start_frame:
					process.stdin.write(b)
					progress_queue.put(1)
		
		elif len(window_buffer) == 1:
			abs_frame_idx = current_read_idx - 1
			if abs_frame_idx >= start_frame:
				sr_tensors = wrapper.Process(window_buffer)
				final_bytes = Worker.tensors_to_bytes_static([sr_tensors[0]])
				process.stdin.write(final_bytes[0])
				progress_queue.put(1)
		
		process.stdin.close()
		process.wait()
		cap.release()
		
	# =====================================================================
	# 主控业务逻辑 (ProcessUpscale)
	# =====================================================================
	
	def ProcessUpscale(self, input_path, output_dir, isfinal):
		# 1. 探针：获取视频参数并使用模型自身的倍率计算分辨率
		import os
		cap = cv2.VideoCapture(input_path)
		org_fps = cap.get(cv2.CAP_PROP_FPS)
		orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		cap.release()
		
		up_w, up_h = self.upscaler.GetSize(orig_w, orig_h)
		base_name = os.path.splitext(os.path.basename(input_path))[0]
		
		# 2. 组装输出路径
		if isfinal:
			target_output = output_dir
		else:
			# 【生命周期管理】作为半成品的中间件坚决不调用 AddTempFile()，留给后续插帧环节接盘
			target_output = os.path.join(output_dir, f"{base_name}_upscaled.mp4")
		
		# 3. 将任务丢给公共分块调度引擎
		result_path = self._run_in_chunks(
			stage=1,
			input_path=input_path,
			output_path=target_output,
			isfinal=isfinal,
			wrapper=self.upscaler,
			out_w=up_w,
			out_h=up_h,
			out_fps=org_fps,
			base_name=base_name,
			desc="放大"
		)
		
		return result_path
	
	def ProcessInterpolate(self, input_path, output_path, isnoupscale):
		if not isnoupscale:
			Utils.AddTempFile(input_path)
		raise NotImplementedError("尚未实现")
	
	def MergeOtherTracks(self, input_path, output_path):
		pass