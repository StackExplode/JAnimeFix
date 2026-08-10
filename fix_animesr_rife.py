import argparse
import subprocess
import cv2
import numpy as np
from tqdm import tqdm
from animesr_wrapper import AnimeSRWrapper

# ==========================================
# 用户可调参数配置区 (Global Configuration)
# ==========================================
DEVICENUM = 1  # 指定使用的 GPU 编号 (0, 1, 2, ...)
DEVICE = f"cuda:{DEVICENUM}"  # 推理设备 ('cuda' 或 'cpu')
ANIMESR_MODEL_PATH = "models/AnimeSR_v2.pth"  # AnimeSR 模型路径
RIFE_MODEL_PATH = "models/rife2.13.pkl"  # RIFE 模型路径

UPSCALING_FACTOR = 2  # 放大倍率 (通常为 2 或 4)
INTERPOLATION_FACTOR = 2  # 插帧倍率 (2, 3, 或 4)
OUTPUT_SHORT_EDGE = 1080  # 最终输出视频的短边分辨率

WINDOW_SIZE = 12  # 滑动窗口大小 (每次存入内存的原视频帧数，包含重叠帧)
BATCH_SIZE = 16  # 模型推理时的并行 Batch Size

# FFmpeg 编码参数配置
FFMPEG_VCODEC = "h264_nvenc"  # 视频编码器 (如 libx264, libx265, h264_nvenc, hevc_nvenc)
FFMPEG_PRESET = "fast"  # 编码速度与压缩率预设 (如 fast, medium, slow)
FFMPEG_CRF = "18"  # 恒定质量因子 (18-23视觉无损，数值越小质量越高，体积越大)


# ==========================================

def load_models():
	"""
	[占位函数] 初始化并加载模型到指定 DEVICE。
	"""
	print(f"正在将模型加载至 {DEVICE}...")
	animesr = AnimeSRWrapper(
		model_path=ANIMESR_MODEL_PATH,
		device=DEVICE,
		outscale=UPSCALING_FACTOR
	)
	rife = None  # 替换为实际的模型实例
	return animesr, rife


def run_animesr_batch(model, frames, batch_size):
	"""
	直接调用包装器处理整个窗口序列。
	由于 AnimeSR 是基于 RNN 的时序模型，它内部自带了序列循环，
	因此无需手动按照 batch_size 切片，直接将整个 window 的帧列表传入即可。
	"""
	upscaled_frames = model.process_sequence(frames)
	return upscaled_frames


def run_rife_batch(model, frames, factor, batch_size):
	"""
	[占位函数] 对传入的高清连续帧列表进行 RIFE 插帧。
	注意：输入 N 帧，会产生间隔，最终输出应该是一条完整的时间线序列。
	"""
	# 伪代码逻辑：如果 factor=2，输入帧 [A, B, C]
	# 需要计算 A-B 之间的插帧，B-C 之间的插帧
	# 最终返回 [A, A.5, B, B.5, C] 这样的完整序列
	
	# 下面仅作占位示意，直接复制原帧模拟生成过程
	interpolated_sequence = []
	for i in range(len(frames) - 1):
		interpolated_sequence.append(frames[i])
		for _ in range(factor - 1):
			interpolated_sequence.append(frames[i])  # 假装这里是算出来的中间帧
	
	# 永远要把最后一帧加上，保证序列闭合
	interpolated_sequence.append(frames[-1])
	return interpolated_sequence


def resize_to_short_edge(image_list, short_edge):
	"""
	将图像列表缩放到指定的短边长度，保持横纵比。
	"""
	if not image_list: return []
	
	h, w = image_list[0].shape[:2]
	if h < w:
		new_h = short_edge
		new_w = int(w * (short_edge / h))
	else:
		new_w = short_edge
		new_h = int(h * (short_edge / w))
	
	return [cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4) for img in image_list]


def process_video(input_path, output_path):
	# 1. 初始化模型
	animesr_model, rife_model = load_models()
	
	# 2. 读取原视频信息
	cap = cv2.VideoCapture(input_path)
	if not cap.isOpened():
		raise ValueError(f"无法打开输入视频: {input_path}")
	
	orig_fps = cap.get(cv2.CAP_PROP_FPS)
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	
	# 3. 计算输出视频属性
	out_fps = orig_fps * INTERPOLATION_FACTOR
	
	# 计算输出分辨率 (基于放大倍率和设定的短边要求)
	# 在这个框架中，我们在经过 AnimeSR 后统一执行一次 Resize
	temp_h, temp_w = orig_h * UPSCALING_FACTOR, orig_w * UPSCALING_FACTOR
	if temp_h < temp_w:
		out_h = OUTPUT_SHORT_EDGE
		out_w = int(temp_w * (OUTPUT_SHORT_EDGE / temp_h))
	else:
		out_w = OUTPUT_SHORT_EDGE
		out_h = int(temp_h * (OUTPUT_SHORT_EDGE / temp_w))
	
	# 确保宽高是偶数（H264 编码要求）
	out_w = out_w if out_w % 2 == 0 else out_w + 1
	out_h = out_h if out_h % 2 == 0 else out_h + 1
	
	print(f"原视频: {orig_w}x{orig_h} @ {orig_fps:.2f}fps, 共 {total_frames} 帧")
	print(f"输出视频: {out_w}x{out_h} @ {out_fps:.2f}fps")
	
	# 4. 启动 FFmpeg Pipe (通过 stdin 接收原始像素流)
	ffmpeg_cmd = [
		'ffmpeg',
		'-y',  # 覆盖输出
		'-f', 'rawvideo',  # 输入格式为原生像素
		'-vcodec', 'rawvideo',
		'-s', f'{out_w}x{out_h}',  # 输入分辨率
		'-pix_fmt', 'bgr24',  # OpenCV 的默认色彩通道顺序
		'-r', str(out_fps),  # 输入及输出帧率
		'-i', '-',  # 从 stdin 接收数据
		'-c:v', FFMPEG_VCODEC,
		'-preset', FFMPEG_PRESET,
		'-cq', FFMPEG_CRF,
		'-pix_fmt', 'yuv420p',  # 转换回常规播放器兼容的色彩空间
		'-gpu', str(DEVICENUM),  # 指定 GPU 编号 (仅对 NVENC 有效)
		output_path
	]
	
	process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
	
	# 5. 主处理循环 (带重叠滑动窗口)
	window_buffer = []
	pbar = tqdm(total=total_frames, desc="处理进度", unit="帧")
	
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		
		window_buffer.append(frame)
		pbar.update(1)
		
		# 当缓冲区满了（达到窗口大小）
		if len(window_buffer) == WINDOW_SIZE:
			# A. 放大
			sr_frames = run_animesr_batch(animesr_model, window_buffer, BATCH_SIZE)
			
			# B. 插帧 (传入放大后的帧)
			interp_frames = run_rife_batch(rife_model, sr_frames, INTERPOLATION_FACTOR, BATCH_SIZE)
			
			# C. 调整到目标短边分辨率
			final_frames = resize_to_short_edge(interp_frames, OUTPUT_SHORT_EDGE)
			
			# D. 写入管道 (注意：不写入窗口产生的最后一帧，因为下一轮它会作为第一帧被再次计算和输出，避免重复帧)
			# 例如 factor=2，输入帧 [0, 1, 2]，输出 [0, 0.5, 1, 1.5, 2]。我们只写到 1.5。
			frames_to_write = len(window_buffer) - 1
			frames_to_write_after_interp = frames_to_write * INTERPOLATION_FACTOR
			
			for img in final_frames[:frames_to_write_after_interp]:
				process.stdin.write(img.tobytes())
			
			# E. 滑动窗口：只保留最后一帧作为新窗口的开头
			window_buffer = [window_buffer[-1]]
	
	# 处理视频末尾剩余的帧
	if len(window_buffer) > 1:
		sr_frames = run_animesr_batch(animesr_model, window_buffer, BATCH_SIZE)
		interp_frames = run_rife_batch(rife_model, sr_frames, INTERPOLATION_FACTOR, BATCH_SIZE)
		final_frames = resize_to_short_edge(interp_frames, OUTPUT_SHORT_EDGE)
		
		# 视频最后一次处理，将所有剩余帧及其插帧（包含真正的最后一帧）全部写入
		for img in final_frames:
			process.stdin.write(img.tobytes())
	
	elif len(window_buffer) == 1:
		# 只剩孤立的一帧（极小概率刚好整除），单纯放大后写入，无法插帧
		sr_frame = run_animesr_batch(animesr_model, window_buffer, BATCH_SIZE)[0]
		final_frame = resize_to_short_edge([sr_frame], OUTPUT_SHORT_EDGE)[0]
		process.stdin.write(final_frame.tobytes())
	
	# 6. 收尾清理
	cap.release()
	pbar.close()
	
	print("正在等待 FFmpeg 完成最后编码...")
	process.stdin.close()
	process.wait()
	print("处理完成！视频已保存至:", output_path)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="流式视频超分与插帧处理脚本")
	parser.add_argument("-i", "--input", required=True, help="输入视频文件路径")
	parser.add_argument("-o", "--output", required=True, help="输出视频文件路径")
	
	args = parser.parse_args()
	process_video(args.input, args.output)