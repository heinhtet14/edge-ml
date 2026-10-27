# Orin Nano Perception Lab: hands-on plan + links

Goal: reproduce the interview pipeline on your own Orin Nano. **Budget → ONNX → TensorRT FP16 → INT8 → validate → coexist with a real-time loop.**
Write down every number (latency p50/p99, FPS, mAP, power, temperature). That becomes real interview evidence.

## Before you start: two Orin Nano facts
- **The Orin Nano has no DLA.** DLA cores exist only on Orin NX and AGX Orin. In interviews, say "offload to DLA *on NX/AGX*".
- **Check your TensorRT version** (`dpkg -l | grep tensorrt`). JetPack 6.x ships TensorRT 10.x. There, the old *implicit* INT8 calibration (IInt8Calibrator) is deprecated, and the modern path is **explicit Q/DQ quantization with NVIDIA Model Optimizer**. Learn both; the explicit path is the one to talk about.

---

## Step 0: Setup and monitoring
- **jtop (jetson-stats):** CPU/GPU/memory/temperature/power, power mode, jetson_clocks. `sudo pip3 install -U jetson-stats`
  https://github.com/rbonghi/jetson_stats · https://pypi.org/project/jetson-stats/
- **jetson-containers (dusty-nv):** prebuilt PyTorch/TensorRT/ROS containers matched to your JetPack (`autotag`)
  https://github.com/dusty-nv/jetson-containers
- **Jetson AI Lab tutorials:** https://jetson-ai-lab.com/tutorial_ultralytics.html
- Fix the power mode (`sudo nvpmodel -q`, then MAXN / MAXN SUPER) and run `sudo jetson_clocks` before benchmarking. Record which mode you used.

## Step 1: Baseline, YOLO → TensorRT FP16 (on device)
- **Ultralytics TensorRT export:** `yolo export model=yolo11n.pt format=engine half=True`
  https://docs.ultralytics.com/integrations/tensorrt · https://docs.ultralytics.com/tasks/export
- **Benchmark the raw engine with trtexec:** `trtexec --onnx=m.onnx --fp16 --saveEngine=m_fp16.engine`, then `trtexec --loadEngine=...` for latency percentiles
  https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/command-line-programs.html
- Compare **trtexec latency (network only)** with **end-to-end latency (pre-processing + network + NMS)**. The gap is the "profile before you quantize" lesson.

## Step 2: INT8
- **Easy path (implicit calibration):** `yolo export model=yolo11n.pt format=engine int8=True data=coco8.yaml` (use more calibration images, ideally your own camera frames). Build on the Orin itself; calibration is device-specific.
- **Modern path (explicit Q/DQ):** NVIDIA Model Optimizer ONNX PTQ (INT8, `max` vs `entropy` calibration), then build with trtexec.
  https://github.com/NVIDIA/TensorRT-Model-Optimizer (see `examples/onnx_ptq`) · docs: https://nvidia.github.io/Model-Optimizer/
- Measure: **mAP (FP32 vs FP16 vs INT8)**, p99 latency, engine size, power.

## Step 3: Debug accuracy drops (layer by layer)
- **Polygraphy:** compare TensorRT vs ONNX Runtime layer outputs to find the first layer that diverges:
  `polygraphy run m.onnx --trt --onnxrt --int8 --trt-outputs mark all --onnx-outputs mark all`
  https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy · https://pypi.org/project/polygraphy/
- TensorRT accuracy troubleshooting: https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/troubleshooting-reporting.html
- Then try mixed precision: keep the sensitive layers (first conv, detection head) in FP16.

## Step 4: Profile the full pipeline
- **Nsight Systems:** a timeline of CPU threads, CUDA kernels and memory copies. Shows whether pre/post-processing or the network dominates.
  https://developer.nvidia.com/nsight-systems
- Move pre-processing to the GPU (resize/normalize in CUDA or VPI) and re-measure.

## Step 5: Structured pruning
- **Torch-Pruning (DepGraph, CVPR 2023):** physically removes channels, so you get real speedups (unlike `torch.nn.utils.prune` masks)
  https://github.com/VainF/Torch-Pruning
- Prune ~30% of channels → fine-tune → re-export → compare mAP and latency with INT8 alone and with pruning + INT8.

## Step 6: Into ROS 2
- **Isaac ROS object detection (`isaac_ros_yolov8`, RT-DETR):** TensorRT inference + NITROS zero-copy on ROS 2
  https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_object_detection · NITROS: https://developer.nvidia.com/blog/boosting-custom-ros-graphs-using-nvidia-isaac-transport-for-ros/
- Note: the YOLOv8 decoder has the class count hard-coded. Edit it for custom models.

## Step 7: The overrun experiment (best interview story)
Recreate the question you didn't know and answer it with data.
1. Run a 1 kHz `sleep_until` loop (like your CAN thread) that counts overruns and logs the worst-case latency.
2. Run the YOLO engine flat out at the same time. **Record the overruns.**
3. Add fixes one at a time and re-measure each:
   - `isolcpus=` in `/boot/extlinux/extlinux.conf` + `taskset`/pinning
   - `SCHED_FIFO` (`chrt -f 80`)
   - `mlockall`
   - fixed clocks
- **cyclictest (rt-tests)** gives standard scheduling-latency numbers: https://wiki.linuxfoundation.org/realtime/documentation/howto/tools/rt-tests
- **RT kernel on Jetson:** NVIDIA's guide (the biggest gains come from RT kernel + CPU isolation + RT scheduling *together*; the Jetson RT kernel is developer-preview quality):
  https://docs.nvidia.com/holoscan/sdk-user-guide/setup/additional-setup/optimizing-jet-pack
  Community walkthrough (JetPack 6.2.1): https://wiki.seeedstudio.com/flash_preempt_rt_kernel_on_recomputer_jetson_jetpack_6_2_1/

## Bonus: for the interviewer's hardware (NX/AGX)
- **DLA tutorial** (PyTorch → ONNX → TensorRT on DLA, INT8, Nsight): https://github.com/NVIDIA-AI-IOT/jetson_dla_tutorial
- Blog: https://developer.nvidia.com/blog/getting-started-with-the-deep-learning-accelerator-on-nvidia-jetson-orin/
- TensorRT "Working with DLA": https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dla.html

---

## Results table to fill in

| Config | Network latency p50/p99 (ms) | End-to-end p99 (ms) | FPS | mAP | Engine MB | Power (W) |
|---|---|---|---|---|---|---|
| PyTorch FP32 | | | | | | |
| TRT FP16 | | | | | | |
| TRT INT8 (implicit) | | | | | | |
| TRT INT8 (ModelOpt Q/DQ) | | | | | | |
| Pruned + INT8 | | | | | | |

| Overrun experiment | Overruns / 10k cycles | Worst-case latency (µs) |
|---|---|---|
| Loop alone | | |
| + YOLO, no isolation | | |
| + isolcpus + pinning | | |
| + SCHED_FIFO + mlockall | | |
