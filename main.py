import argparse
import json

from utils import Utils
from worker import Worker

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input", required=True)
	parser.add_argument("-o", "--output", required=True)
	args = parser.parse_args()
	
	configpath = "config/global.json"
	with open(configpath, "r") as f:
		gconfig = json.load(f)
	
	upscaler_name = gconfig.get("upscaler", "none")
	if upscaler_name != "none":
		with open(f"config/{upscaler_name}.json", "r") as f:
			uconfig = json.load(f)
		upscaler = Utils.CreateInstance(f"driver.{upscaler_name}.Wrapper_{upscaler_name}.Wrapper_{upscaler_name}", gconfig, uconfig)
	else:
		upscaler = None
	
	vfi_name = gconfig.get("interpolator", "none")
	if vfi_name != "none":
		with open(f"config/{vfi_name}.json", "r") as f:
			vconfig = json.load(f)
		interpolator = Utils.CreateInstance(f"driver.{vfi_name}.Wrapper_{vfi_name}.Wrapper_{vfi_name}", gconfig, vconfig)
	else:
		interpolator = None
	
	print(f"开始进行视频处理，使用设备：{gconfig.get('device', 'cpu')}...")
	worker = Worker(gconfig, upscaler, interpolator)
	#worker.process_video(args.input, args.output)
	Utils.InitTempDir(gconfig)
	tempdir = gconfig.get("temp_dir")
	interpo_input = args.input
	
	if upscaler is not None:
		print("开始进行放大处理...")
		isfinish = interpolator is None
		outputdir =  args.output if isfinish else tempdir
		interpo_input = worker.ProcessUpscale(args.input, outputdir, isfinish)
	if interpolator is not None:
		print("开始进行插帧处理...")
		worker.ProcessInterpolate(interpo_input, args.output)
		
	print("开始合并其他轨道...")
	worker.MergeOtherTracks(args.input, args.output)
	
	print("清理临时文件...")
	Utils.CleanupTempFiles()