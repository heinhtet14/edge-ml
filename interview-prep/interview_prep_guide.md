# Interview Prep: Senior ML for Robotics Engineer – Edge AI Systems (Abu Dhabi)

**Round 1:** Technical conversation with Hiring Manager, Prof. Andrea Finazzi
**When:** Thursday 29 Oct 2026, 7:00 pm. Check the time zone (Abu Dhabi is GST, UTC+4).
**Format:** 1 hour, conversational, no live exercises. He'll explain the role, so plan questions to ask.
**Next round:** a team interview OR a take-home assessment, then HR (visa/relocation).

---

## 1. About the interviewer (verify this)

I couldn't confirm a public faculty profile for "Prof. Andrea Finazzi" in Abu Dhabi. One plausible match:
- **Andrea Finazzi**: B.S. Politecnico di Milano (Computer Systems Eng.), then M.S./research at **KAIST USRG**
  (Prof. David Hyunchul Shim's Unmanned Systems lab). Before that, 2 years as a software developer (Dedagroup, Milan).
- He co-authored KAIST's **autonomous racing** work for the Indy Autonomous Challenge:
  - *An autonomous system for head-to-head race: Team KAIST at the IAC* (arXiv 2303.09463)
  - *Resilient navigation and path planning system for high-speed autonomous race car* (arXiv 2207.12232)
  - *Game-theoretic MPC with data-driven vehicle model identification for head-to-head racing* (arXiv 2106.04094)
- There is an Abu Dhabi link: KAIST competed in A2RL (Abu Dhabi Autonomous Racing League).

**If it's him:** he's a hands-on systems person. Expect him to care about real-time performance on real hardware, state
estimation, robustness when things go wrong, and running everything on limited onboard compute. Skim the abstract and
system diagram of 2303.09463 before Thursday. Don't overclaim that you know his work. Something like *"I read your
team's IAC system paper, and the way you handled X was interesting"* is enough.
Check his LinkedIn to confirm his current role and the client company.

---

## 2. What the role really is (in one sentence)

> Own the **whole ML stack on a humanoid robot**: choose models, optimize them (quantize/prune/TensorRT), and run them
> **reliably in real time on NVIDIA Jetson** alongside robotics and control processes, then lead the team that does it.

Keywords they'll listen for: *end-to-end ownership, trade-offs, Jetson/TensorRT, real-time, C++ & Python, ROS 2,
classic ML vs DL judgment, leadership.*

---

## 3. Your story (prepare these before anything else)

> Full project details and the CV-vs-code claim-safety table are in **`project_refresher.md`**.

### 90-second intro (rehearse aloud, adapt to your own words)
> "I'm Hein, an edge ML engineer with about four years of taking models from training to real-time execution on
> constrained hardware. Most relevant here: at Magick Robotics I led the team building Tet Hlan, a 12-DOF bipedal
> humanoid. I trained the locomotion policy with PPO in Isaac Lab and deployed it on a Jetson Orin Nano at 50 Hz,
> and I wrote the C++ real-time motor stack underneath: 12 CubeMars motors on two CAN buses, one control thread per
> bus. We reached stable standing balance on hardware, and I learned a lot about the sim-to-real gap.
> Before that I ported ORB-SLAM2's numerical core to dependency-free C at Panasonic for constrained compute.
> Now I'm AI Architecture Lead at Magick Mind, co-leading about 10 engineers, where I built the real-time voice
> runtime and an LLM-to-robot interface for our devices, which taught me a lot about edge-vs-cloud splits.
> This role pulls those threads together: owning the ML stack on a humanoid, on Jetson, end to end. That's exactly
> what I want to do next."

### STAR stories (Situation, Task, Action, Result)
| Theme | Your story |
|---|---|
| Edge deployment end to end | Tet Hlan: Isaac Lab PPO → TorchScript/ONNX → Orin Nano 50 Hz multi-process node → C++ CAN stack → standing balance |
| Debugging | "Actions too high": unbounded RSL-RL actor + no normalizer/clipping → added action stats, analysis script, clipping, sim2sim joint-limit clamps |
| Real-time design trade-off | One 1 kHz thread per CAN bus, lock-free atomics, absolute-deadline loop with overrun counting; ankle LUT instead of an iterative IK solver in the RT loop |
| Edge vs cloud / latency | Magick Mind: dual-brain reflex/fast/smart routing; VAD on server gating paid sessions; devices hold no keys (token broker) |
| Classic vs neural, with data | Raggity: BM25 + cross-encoder matched dense quality with no embedder; profiling showed the embedder was 94% of latency |
| Leadership | Tet Hlan cross-disciplinary team lead; Magick Mind ~10 engineers: contract-first protos, design docs, consolidating 4 persona services into 1 |
| Safety for embodied AI | "The model picks the verb, the robot owns the motion": capability manifest, LLM params ignored, numeric limits, preemption, e-stop |
| Failure / what you'd do differently | Hardware only reached standing: no gain randomization/pushes, unmodeled latency, no RT scheduling → the concrete next-step list in project_refresher §1.6 |

### Questions about your gaps (answer calmly, don't apologize)
- **"You have a B.Eng; we asked for MSc/PhD."** → "My depth comes from shipping on real hardware: a humanoid from RL
  training to motor control, and SLAM math from scratch in C. I'm happy to go as deep as you want on any of it."
- **"TensorRT/INT8 experience?"** → Be honest: your deployed models were small MLPs where ORT/TorchScript was the
  right call. Then show you know the pipeline (project_refresher §5) and the bandwidth/memory reasoning (edge-llm-lab).
- **"CPU isolation / RT kernel?"** → "Not yet on Tet Hlan. We used absolute-deadline loops and overrun counters. Next
  would be SCHED_FIFO, isolcpus + pinning, mlockall, PREEMPT_RT, and moving inference into C++."

For each one, be ready to go **2–3 levels deeper**: "Why that approach? What else did you try? What broke?"

---

## 4. Technical topics to talk about fluently

### 4.1 NVIDIA Jetson platform
- **Family:** Orin Nano / Orin NX / **AGX Orin** (Ampere GPU, up to ~275 INT8 TOPS, 32–64 GB); **Jetson Thor**
  (Blackwell, aimed at humanoids, FP4/FP8 support). Know what you've actually used.
- **Unified memory:** CPU and GPU share the same DRAM, so zero-copy / pinned buffers avoid cudaMemcpy. But memory
  bandwidth is shared, so heavy GPU load can slow CPU-side control processes.
- **Accelerators:** GPU, plus **DLA** (Deep Learning Accelerator: low power, INT8/FP16, limited ops, falls back to GPU)
  and **PVA/VIC** for vision pre-processing. Offloading a model to DLA frees the GPU for other work.
- **Tooling:** JetPack/L4T versions, `nvpmodel` (power modes), `jetson_clocks`, `tegrastats`, Nsight Systems, `trtexec`.
- **Thermal/power:** sustained vs burst performance. Design for the thermally throttled steady state. Power budget
  matters on a battery-powered humanoid.

### 4.2 Model optimization pipeline
`PyTorch → ONNX export → (graph surgery / simplify) → TensorRT engine (FP16/INT8) → validate accuracy + latency on target → deploy`
- **TensorRT:** layer/kernel fusion, precision selection, kernel auto-tuning. **Engines are tied to the GPU + TRT
  version**, so build them on the target (or a matching device) and use a timing cache. Dynamic shapes need
  optimization profiles. Unsupported ops need custom plugins or a model change.
- **Quantization:**
  - PTQ (calibration: entropy/min-max/percentile, representative data) vs **QAT** (when PTQ loses too much accuracy).
  - Per-channel weights, per-tensor activations. Keep sensitive layers (first/last, softmax, LayerNorm, detection heads) in FP16 (mixed precision).
  - Do per-layer sensitivity analysis. Explicit Q/DQ nodes (from NVIDIA's ModelOpt / pytorch-quantization).
  - LLM/VLM: weight-only INT4 (AWQ, GPTQ), KV-cache size, TensorRT-LLM / llama.cpp / vLLM on Jetson.
- **Pruning:** structured (channels/heads, gives real speedup) vs unstructured (needs sparse hardware support).
  **2:4 structured sparsity** is supported by Ampere tensor cores (Orin). Prune, then fine-tune.
- **Distillation:** a large teacher trains a small student. Often the best accuracy-per-millisecond.
- **Runtime:** CUDA streams, CUDA Graphs (lower launch overhead), batching across cameras, async pre/post-processing,
  GPU pre-processing (avoid CPU resize/normalize), preallocated buffers, no allocations in the hot loop.
- **Always measure:** p50/**p99** latency and jitter (not just the mean), throughput, power (W), memory, and accuracy delta vs FP32.

### 4.3 Real-time systems & embedded Linux (the "systems" half of the JD)
- **CPU isolation:** `isolcpus`/`nohz_full`, cgroups/cpusets, `taskset`, IRQ affinity, `SCHED_FIFO` priorities,
  **PREEMPT_RT** kernel for control loops. Keep ML inference off the cores that run the control loop.
- **Multi-process coordination:** shared memory, ROS 2 zero-copy (iceoryx, loaned messages), NVIDIA Isaac ROS
  **NITROS** (GPU-buffer type adaptation), lock-free queues, watchdogs, health monitoring.
- **Containerization:** L4T-based images, NVIDIA container runtime, matching JetPack/CUDA/TRT versions, reproducible
  builds, OTA updates, rollback.
- **Reliability:** graceful degradation (fall back to a smaller model or a safe behavior), timeouts, deadline-miss
  monitoring, logging/telemetry, deterministic startup.

### 4.4 Edge vs cloud trade-offs
- **Must stay onboard:** anything in the safety or control loop (balance, collision avoidance, state estimation, reflexes).
  Network latency and dropouts are unacceptable there.
- **Can go to the cloud/offboard:** fleet learning, data logging/labeling, retraining, heavy VLM/LLM reasoning that is
  slow and asynchronous (task planning), map building.
- Talk about it as a **hierarchy**: fast loop (1 kHz joint control, ~50–100 Hz policy) → perception (10–30 Hz) →
  slow reasoning (≤1–5 Hz, possibly offboard). This is similar to the "System 1 / System 2" design in humanoid VLA
  stacks (e.g. Figure's Helix, NVIDIA GR00T N1).

### 4.5 Traditional ML vs deep learning (a JD favorite: have a decision framework)
Choose **classic ML / model-based** when data is small, you need interpretability or determinism, compute is tiny, or the
physics is well known. Examples: Kalman/EKF/UKF filters, ICP, GMM/HMM, SVM/random forest/gradient boosting for terrain or
contact classification, anomaly detection on joint telemetry, gait-phase estimation.
Choose **deep learning** for high-dimensional perception (vision, depth, segmentation), when you have lots of data,
when you need generalization (VLMs, open-vocabulary detection), and for learned locomotion policies (RL).
Choose **hybrid** (often best): learned perception feeding a classic estimator/controller, or a learned residual on
top of a model-based controller.
Criteria to state out loud: *accuracy, latency, power, data availability, robustness/OOD behavior, verifiability,
maintainability.*

### 4.6 Robotics fundamentals (expect light probing)
- **ROS 2:** DDS, QoS profiles (reliable vs best-effort for sensor data), executors and callback groups, composable
  nodes (intra-process zero-copy), lifecycle nodes, `ros2_control`, `tf2`, time sync.
- **Sensors:** IMU, joint encoders, force/torque sensors, stereo/RGB-D, LiDAR; calibration, timestamp synchronization.
- **Control:** PID, MPC, whole-body control, impedance control. Loop rates and why they matter.
- **Kinematics:** forward/inverse kinematics, Jacobians, singularities.
- **Perception:** detection (YOLO family, RT-DETR), segmentation, 3D/BEV, visual(-inertial) SLAM (ORB-SLAM3, Isaac
  cuVSLAM), sensor fusion with an EKF.
- **Humanoid learning:** RL locomotion in simulation (Isaac Lab/MuJoCo), domain randomization, sim-to-real gap,
  imitation learning from teleoperation, VLA models (OpenVLA, π0, GR00T).

### 4.7 C++ / Python
- C++: RAII, move semantics, smart pointers, avoiding allocations/locks in real-time paths, lock-free ring buffers,
  templates vs virtual dispatch cost, profiling (perf, Nsight).
- Python ↔ C++: pybind11. Prototype in Python, deploy hot paths in C++.
- Code review: what you look for (ownership/lifetimes, thread safety, error handling, tests, measurable performance).

---

## 5. Practice the system-design talk-through (likely, even in a "conversational" round)

**Prompt:** *"How would you design the perception + ML stack for our humanoid on a Jetson AGX Orin?"*

Answer structure:
1. **Clarify:** tasks (navigation? manipulation? human interaction?), sensors, latency targets, power budget,
   what already exists.
2. **Budget:** compute and latency per module (e.g. control 1 kHz on isolated CPU cores; locomotion policy 50 Hz;
   detection/segmentation 15–30 Hz on GPU; depth on DLA/VPI; VLM at 1–2 Hz or offboard).
3. **Pipeline:** camera → GPU pre-processing (zero-copy) → TensorRT models (FP16/INT8) → tracking/fusion (EKF) →
   world model → planner/controller via ROS 2 (NITROS/zero-copy).
4. **Optimization strategy:** profile first, then quantize, prune/distill, use DLA offload, CUDA Graphs.
5. **Reliability:** watchdogs, fallbacks, thermal monitoring, deadline-miss metrics, containerized deployment + OTA.
6. **Validation:** accuracy on the target hardware, HIL tests, regression benchmarks in CI (latency/accuracy gates),
   field logging into a data flywheel.
7. **Trade-offs:** say what you'd give up and why.

Practice this aloud twice, about 5 minutes each.

---

## 6. Questions to ask him (pick 5–6)
1. What does the humanoid platform look like today: hardware, Jetson model (Orin or Thor), sensors, OS/ROS 2 setup?
2. What's on the ML stack right now, and what's the biggest bottleneck: latency, accuracy, reliability, or tooling?
3. What would success look like in the first 3 / 6 / 12 months?
4. How big is the team, and how is it split across ML, robotics/control, and infrastructure? Who would I mentor?
5. Are you using learned policies (RL/VLA) for locomotion and manipulation, or classic control? Where does onboard
   compute end and offboard compute begin?
6. How do you validate models before they go on the robot (sim, HIL, CI benchmarks)?
7. Is the center research-oriented or product-oriented? Do you publish?
8. What does the next stage look like: team interview or assessment? What will it focus on?

---

## 7. Remaining days
| When | Do |
|---|---|
| **Tue (today)** | Write the 90-sec intro + STAR stories with metrics. Confirm the interviewer on LinkedIn. |
| **Wed** | Review sections 4.1–4.5 out loud. Skim KAIST IAC paper (2303.09463). Practice section 5 twice. |
| **Thu daytime** | Light review only. Test camera/mic/internet, quiet room, CV + JD open, water. |
| **Thu 7pm** | Be conversational. Think aloud through trade-offs. Ask your questions. Ask about next steps. |

**Tips:** use numbers whenever you can. Say "I don't know, but here's how I'd find out / reason about it" instead of
bluffing. Bring every answer back to *ownership of the full stack* and *running on real hardware*.

---

## 8. Mock interview takeaways (27 Oct)

**Strengths:** detailed Tet Hlan knowledge. Strong stories: servo→MIT mode + LiDAR state estimation; body-frame velocity rotation.

**Fix these:**
1. **Answers too short.** Aim for 45–90 s. Structures:
   - System question: follow the data, sensors → observation → policy → actions → motors, and say how the pieces talk to each other.
   - "How would you…": budget → pipeline → validate → live alongside control.
   - Debugging or "what would you do": measure first → hypotheses ranked → fix → staged test.
2. **Use your evidence.** Edge vs cloud → dual brain, "the model picks the verb, the robot owns the motion", VAD gating. Leadership → Tet Hlan lead, ~10 engineers, consolidation decisions.
3. **Gap questions:** "I haven't hit this exact case, but here's how I'd reason…" (hypotheses → how to confirm → fix). Never "I promise I'm dedicated". On the degree: show evidence that you lead; don't frame yourself as junior.

**Key answers to say aloud (2–3× each):**
- **Intro:** 5 beats + why this role.
- **Robot loop:** 200 Hz IMU (I2C) + CAN feedback → 44-dim obs (ankle FK via LUT, closed kinematic chain) → 50 Hz multi-process policy → atomics → 1 kHz thread per bus → MIT frames at 200 Hz.
- **FAST-LIO velocity:** iterated EKF, IMU propagation ~200 Hz + LiDAR correction 10 Hz → rotate to body frame with `R_wbᵀ` (lever arm `ω × r`). Check which IMU you used.
- **Why only standing:** actuator + latency gap, 3D-printed links (mass/CoM/flex), no gain randomization/pushes. Plan: month 1 measure (logs, latency, weigh links, actuator net), month 2 retrain (latency/gain/push DR, C++ inference, RT scheduling), month 3 staged tests on a gantry.
- **Perception on Orin:** budget → profile (pre/post-processing!) → ONNX → TRT FP16 on device → INT8 with robot-camera calibration → check calibration → mixed precision on sensitive layers → QAT → prune/smaller model. Validate mAP, p99, power after heat-up.
- **1 kHz overruns with detector:** CPU contention (normal scheduler), shared-DRAM bandwidth, thermal throttling, IRQs. Confirm: line up overruns with inference frames, check tegrastats/htop/cyclictest. Fix: isolcpus + pinning, SCHED_FIFO, IRQ affinity, mlockall, fixed clocks, PREEMPT_RT.
- **Edge vs cloud:** split by "what if the answer is late or never comes". Offboard sets goals only; robot enforces limits locally.
