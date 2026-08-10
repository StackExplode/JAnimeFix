import argparse
import subprocess
import cv2
import numpy as np
import torch
import queue
import threading
from tqdm import tqdm
from animesr_wrapper import AnimeSRWrapper

# ==========================================
# 配置区
# ==========================================
DEVICENUM = 1  # 指定使用的显卡编号
DEVICE = f"cuda:{DEVICENUM}"
ANIMESR_MODEL_PATH = "models/AnimeSR_v2.pth"
RIFE_MODEL_PATH = "models/rife2.13.pkl"

UPSCALING_FACTOR = 2
INTERPOLATION_FACTOR = 2
OUTPUT_SHORT_EDGE = 1080
WINDOW_SIZE = 24
BATCH_SIZE = 32

FFMPEG_VCODEC = "hevc_nvenc"  # 已改为硬件编码
FFMPEG_PRESET = "p4"  # NVENC 的 preset (p1-p7, p4 为平衡)
FFMPEG_CRF = "18"  # NVENC 中用 -cq 代替 -crf 控制质量


# ==========================================

def load_models():
	print("正在加载模型环境...")
	animesr = AnimeSRWrapper(model_path=ANIMESR_MODEL_PATH, device=DEVICE, outscale=UPSCALING_FACTOR)
	rife = None  # 等待后续 RIFE 封装
	return animesr, rife


def rife_placeholder(tensor_list, factor):
	"""
	正确的 RIFE 占位符函数 (处理 GPU Tensor)。
	通过复制帧来补齐数量，确保输出总帧数与 FFmpeg 设置的 fps 完美匹配。
	"""
	if not tensor_list:
		return []
	
	interp_tensors = []
	# 两两遍历，中间补齐
	for i in range(len(tensor_list) - 1):
		interp_tensors.append(tensor_list[i])
		# 根据插帧倍率复制原帧来占位（如 2 倍插帧则复制 1 次，补充满中间的空隙）
		for _ in range(factor - 1):
			interp_tensors.append(tensor_list[i])
	
	# 永远加上序列的最后一帧闭合
	interp_tensors.append(tensor_list[-1])
	
	return interp_tensors

# --- 线程 1：读取线程 ---
def video_reader_thread(cap, read_queue):
	while True:
		ret, frame = cap.read()
		if not ret:
			read_queue.put(None)  # 放入结束信号
			break
		read_queue.put(frame)


# --- 线程 3：写入线程 ---
def video_writer_thread(process, write_queue):
	while True:
		frame_bytes = write_queue.get()
		if frame_bytes is None:
			break
		process.stdin.write(frame_bytes)


# --- 显卡到内存的转换 (放到最后一步) ---
def tensors_to_bytes(tensor_list, short_edge):
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


def process_video(input_path, output_path):
	animesr_model, rife_model = load_models()
	
	cap = cv2.VideoCapture(input_path)
	orig_fps = cap.get(cv2.CAP_PROP_FPS)
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	orig_w, orig_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	out_fps = orig_fps * INTERPOLATION_FACTOR
	
	temp_h, temp_w = orig_h * UPSCALING_FACTOR, orig_w * UPSCALING_FACTOR
	out_h = OUTPUT_SHORT_EDGE if temp_h < temp_w else int(temp_h * (OUTPUT_SHORT_EDGE / temp_w))
	out_w = int(temp_w * (OUTPUT_SHORT_EDGE / temp_h)) if temp_h < temp_w else OUTPUT_SHORT_EDGE
	out_w, out_h = out_w if out_w % 2 == 0 else out_w + 1, out_h if out_h % 2 == 0 else out_h + 1
	
	ffmpeg_cmd = [
		'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
		'-s', f'{out_w}x{out_h}', '-pix_fmt', 'bgr24', '-r', str(out_fps),
		'-i', '-', '-c:v', FFMPEG_VCODEC, '-preset', FFMPEG_PRESET, '-cq', FFMPEG_CRF,
		'-pix_fmt', 'yuv420p', '-gpu', str(DEVICENUM), output_path
	]
	process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
	
	# 建立多线程队列
	read_queue = queue.Queue(maxsize=30)
	write_queue = queue.Queue(maxsize=30)
	
	# 启动工作线程
	threading.Thread(target=video_reader_thread, args=(cap, read_queue), daemon=True).start()
	writer_t = threading.Thread(target=video_writer_thread, args=(process, write_queue), daemon=True)
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
			
			if len(window_buffer) == WINDOW_SIZE:
				# 1. AnimeSR 放大 (返回 GPU Tensors)
				sr_tensors = animesr_model.process_sequence_tensor(window_buffer)
				
				# 2. RIFE 占位插帧 (真实增加帧数，对齐输出 fps)
				interp_tensors = rife_placeholder(sr_tensors, INTERPOLATION_FACTOR)
				
				# 3. 截断最后的重复帧
				# 原本窗口内要输出的帧数为：len(window_buffer) - 1
				# 插帧后，这部分对应的实际帧数需要乘以插帧倍率
				frames_to_write = len(window_buffer) - 1
				frames_to_write_after_interp = frames_to_write * INTERPOLATION_FACTOR
				
				# 4. 截断后下放到 CPU 并送入写入队列
				final_bytes = tensors_to_bytes(interp_tensors[:frames_to_write_after_interp], OUTPUT_SHORT_EDGE)
				for b in final_bytes:
					write_queue.put(b)
				
				window_buffer = [window_buffer[-1]]
		
		# 处理视频末尾剩余的收尾帧
		if len(window_buffer) > 1:
			sr_tensors = animesr_model.process_sequence_tensor(window_buffer)
			# 同样需要经过占位插帧扩充数量
			interp_tensors = rife_placeholder(sr_tensors, INTERPOLATION_FACTOR)
			final_bytes = tensors_to_bytes(interp_tensors, OUTPUT_SHORT_EDGE)
			for b in final_bytes:
				write_queue.put(b)
		
		elif len(window_buffer) == 1:
			# 只剩孤立的一帧，只能放大，无法进行“两两之间”的插帧
			sr_tensors = animesr_model.process_sequence_tensor(window_buffer)
			final_bytes = tensors_to_bytes([sr_tensors[0]], OUTPUT_SHORT_EDGE)
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


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input", required=True)
	parser.add_argument("-o", "--output", required=True)
	args = parser.parse_args()
	process_video(args.input, args.output)