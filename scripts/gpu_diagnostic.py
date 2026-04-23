import os
import sys
import time
import subprocess
import platform
from collections import deque

# Attempt to import optional dependencies
try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import GPUtil
except ImportError:
    GPUtil = None

try:
    import pyopencl as cl
except ImportError:
    cl = None

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
except ImportError:
    plt = None


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(title):
    print(f"\n{bcolors.HEADER}{bcolors.BOLD}{'='*60}")
    print(f"{title.center(60)}")
    print(f"{'='*60}{bcolors.ENDC}")


def check_system_info():
    print_header("SYSTEM INFORMATION")
    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Processor: {platform.processor()}")
    
    if psutil:
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        print(f"RAM: {vm.total / (1024**3):.2f} GB Total | {vm.available / (1024**3):.2f} GB Available ({vm.percent}% used)")
        print(f"Swap: {sm.total / (1024**3):.2f} GB Total | {sm.free / (1024**3):.2f} GB Free ({sm.percent}% used)")
    else:
        print(f"{bcolors.WARNING}psutil not installed. Skipping RAM details.{bcolors.ENDC}")


def check_pytorch():
    print_header("PYTORCH DIAGNOSTIC")
    if torch:
        print(f"PyTorch Version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {bcolors.OKGREEN if cuda_available else bcolors.FAIL}{cuda_available}{bcolors.ENDC}")
        if cuda_available:
            print(f"GPU Count: {torch.cuda.device_count()}")
            print(f"Current Device: {torch.cuda.current_device()}")
            print(f"Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print(f"{bcolors.FAIL}PyTorch not installed.{bcolors.ENDC}")


def check_tensorflow():
    print_header("TENSORFLOW DIAGNOSTIC")
    if tf:
        print(f"TensorFlow Version: {tf.__version__}")
        cuda_built = tf.test.is_built_with_cuda()
        gpu_devices = tf.config.list_physical_devices('GPU')
        print(f"Built with CUDA: {bcolors.OKGREEN if cuda_built else bcolors.FAIL}{cuda_built}{bcolors.ENDC}")
        print(f"GPUs Visible: {len(gpu_devices)}")
        for i, gpu in enumerate(gpu_devices):
            print(f"  [{i}] {gpu.name}")
    else:
        print(f"{bcolors.FAIL}TensorFlow not installed.{bcolors.ENDC}")


def check_nvidia_smi():
    print_header("NVIDIA-SMI DETAILS")
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        for i, line in enumerate(lines):
            name, total, used, util = line.split(', ')
            print(f"GPU {i}: {bcolors.BOLD}{name}{bcolors.ENDC}")
            print(f"  Usage: {used} / {total} MiB ({util}% Load)")
    except Exception:
        print(f"{bcolors.FAIL}nvidia-smi not found or failed to execute.{bcolors.ENDC}")


def check_amd_opencl():
    print_header("AMD / OPENCL DIAGNOSTIC")
    if cl:
        try:
            platforms = cl.get_platforms()
            if not platforms:
                print("No OpenCL platforms found.")
                return
            for p in platforms:
                print(f"Platform: {p.name} ({p.vendor})")
                devices = p.get_devices()
                for d in devices:
                    d_type = cl.device_type.to_string(d.type)
                    print(f"  Device: {d.name} [{d_type}]")
                    if d_type == 'GPU':
                        print(f"    Global Memory: {d.global_mem_size / (1024**2):.2f} MB")
        except Exception as e:
            print(f"OpenCL Error: {e}")
    else:
        print(f"{bcolors.WARNING}pyopencl not installed.{bcolors.ENDC}")


def monitor_gpu_matplotlib():
    if not plt:
        print(f"{bcolors.FAIL}Matplotlib not installed. Cannot run monitor.{bcolors.ENDC}")
        return

    print(f"{bcolors.OKBLUE}Starting Matplotlib Monitor... (Close window to stop){bcolors.ENDC}")
    
    RECENT_VALUES_COUNT = 30
    history = {'time': deque(maxlen=RECENT_VALUES_COUNT), 'usage': deque(maxlen=RECENT_VALUES_COUNT)}
    
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'r-', label="GPU Usage %")
    ax.set_ylim(0, 100)
    ax.set_title("Real-time GPU Load")
    ax.set_ylabel("Utilization (%)")
    
    def update(i):
        try:
            res = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                 stdout=subprocess.PIPE, text=True).stdout.strip()
            val = float(res.split('\n')[0])
            history['time'].append(time.time())
            history['usage'].append(val)
            line.set_data(range(len(history['usage'])), history['usage'])
            ax.set_xlim(0, len(history['usage']))
            return line,
        except: return line,

    ani = animation.FuncAnimation(fig, update, interval=1000)
    plt.show()


def run_all_diagnostics():
    check_system_info()
    check_pytorch()
    check_tensorflow()
    check_nvidia_smi()
    check_amd_opencl()
    print("\n" + "="*60)
    print("Diagnostic Complete.")
    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Consolidated GPU and System Diagnostic Tool")
    parser.add_argument("--monitor", action="store_true", help="Launch real-time Matplotlib monitor")
    args = parser.parse_args()

    if args.monitor:
        monitor_gpu_matplotlib()
    else:
        run_all_diagnostics()
        print(f"Tip: Run with {bcolors.BOLD}--monitor{bcolors.ENDC} for real-time tracking.")
