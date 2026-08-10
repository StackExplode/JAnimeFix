import argparse
import json

from utils import Utils
from wrapper import WrapperBase
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
		upscaler = Utils.CreateInstance(f"wrapper.Wrapper_{upscaler_name}.Wrapper_{upscaler_name}", gconfig, uconfig)
	else:
		upscaler = WrapperBase(gconfig, {})
	
	vfi_name = gconfig.get("interpolator", "none")
	if vfi_name != "none":
		with open(f"config/{vfi_name}.json", "r") as f:
			vconfig = json.load(f)
		interpolator = Utils.CreateInstance(f"wrapper.Wrapper_{vfi_name}.Wrapper_{vfi_name}", gconfig, vconfig)
	else:
		interpolator = WrapperBase(gconfig, {})
	
	worker = Worker(gconfig, upscaler, interpolator)
	worker.process_video(args.input, args.output)