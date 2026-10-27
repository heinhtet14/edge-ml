# Project Refresher: your own work, ready for deep questions

The companion to `interview_prep_guide.md`. Every fact below was checked against your repos.
**Read section 0 first.** It lists where your CV and your code disagree. Being precise there builds credibility; being vague costs you.

---

## 0. Claim-safety map (CV vs code)

| CV claim | Repo says | How to say it |
|---|---|---|
| 50 Hz policy on Orin Nano via TorchScript / ONNX Runtime | ✅ `rl_controller_twoLeg.py` supports both, 50 Hz steady timer | Say it as is |
| 1 kHz thread per CAN bus | ✅ 1 kHz loop, but **commands go out every 5th cycle = 200 Hz** | "1 kHz control thread per bus, motor commands at 200 Hz, overruns counted" |
| Lock-free atomic targets | ✅ CAN-sub node. ⚠️ pub node uses a try-lock | "Command path is lock-free; feedback path uses try-lock and drops the frame" |
| Ruckig-smoothed commands | ⚠️ **Ankles only**, limits effectively unconstrained | "Ruckig on the parallel-linkage ankles" |
| Domain randomization | ✅ friction/mass/armature/joint friction. ⚠️ **Kp/Kd randomization and pushes commented out** | Say what was on. Present gains/pushes as the next step |
| Walking in sim | ⚠️ Command fixed at **lateral vy = −1.0** (x = 0, yaw = 0) | "Trained a lateral-walking gait in sim" |
| FAST-LIO + MID-360 | Not in these repos | Be ready to explain from memory, or say it was a separate integration |
| CPU isolation / RT scheduling | ❌ None (no SCHED_FIFO, affinity, isolcpus, mlockall) | **Don't claim it.** Explain what you'd add (section 1.6) |
| 50 concurrent voice sessions | ✅ cascade backend only. Gemini default = 4, Cartesia Ink 2 = 8 (vendor quotas) | "Designed for 50; real caps are set by vendor quotas" |
| ~500 ms time-to-first-audio | ⚠️ README claim. Design doc measured **≈1.7 s end-to-end** (1.2 s TTFT + 0.56 s TTS) | Know which segment 500 ms covers. Safer: "~500 ms TTS-side first audio; ~1.7 s end-of-speech to audio, and here's how I'd cut it" |
| Unitree Go2 in MuJoCo | Not found on main (maybe `poc/zenoh` branch) | Check the branch before Thursday |
| TensorRT / quantization / pruning | ❌ Not your code. The `Deep-Learning-Models-Optimization...` folder is **Ahmed Gamal's graduation project** (Jetson Nano) | Don't present it as yours. Use it as study material (section 5) |
| edge-llm-lab | Design + scaffolding only, **no results** | "In progress: designed a reproducible quantized-SLM benchmark for Orin Nano" |

---

## 1. Tet Hlan bipedal humanoid (your strongest story; expect 20+ minutes on it)

### 1.1 One-minute version
> "I led a small team building a 12-DOF biped. I trained a PPO locomotion policy in Isaac Lab, 4096 parallel envs,
> and deployed it on a Jetson Orin Nano at 50 Hz. Under it I wrote the C++ real-time motor stack: 12 CubeMars motors on
> two 1 Mbit/s CAN buses in MIT mode, one thread per bus, plus the IMU node and ankle kinematics. Hardware reached
> stable standing balance. Closing the sim-to-real gap to walking is where I learned the most."

### 1.2 System diagram (draw this if you can share a screen)
```
BNO055 IMU (I2C /dev/i2c-7, 200 Hz) ──┐
                                      ▼
CAN reader threads ──► can_pub node (200 Hz): FK ankle motors→pitch/roll, temp/error checks
                                      │  /tethlan/observations (44 floats)
                                      ▼
        rl_controller (Python, 50 Hz, 3 processes: service / obs receiver / policy executor)
                                      │  /tethlan/actions (12, degrees)
                                      ▼
can_sub node: 2 std::threads @1 kHz (one per bus) ── atomics + sequence counter ── IK + Ruckig (ankles)
                                      │  MIT frames every 5 ms
                                      ▼
            CAN0 / CAN2 @ 1 Mbit/s → 12 CubeMars AK motors (motor runs its own PD)
Launch: layered TimerAction start (IMU 0 s → CAN 2 s → policy 4 s → joystick 5 s), ros2 bag → MCAP, Foxglove
```

### 1.3 RL training facts (`tethlan_reinforcement`)
- **Stack:** Isaac Lab external extension, RSL-RL PPO. Task `Isaac-Tethlanv2-Flat-v0` (rough variant also exists).
- **Joints (12):** hip pitch/roll/yaw, knee, ankle pitch/roll × 2.
- **Observation (44):**
  - base lin vel 3, ang vel 3, projected gravity 3, command 3
  - joint pos (rel. default) 12, **joint vel 8: ankles excluded**, last action 12
  - Noise per group: joint pos ±0.03 hips/knee, ±0.05 roll/yaw, ±0.08 ankles; vel ±1.5. No height scan.
- **Action:** 12 position targets, `target = default + 0.5·a`. **No clipping in training.**
- **PD in sim (Kp/Kd):** hip pitch + knee 10/1.0 (54 Nm), hip roll/yaw 4/0.4, ankles 8/0.8 (16.8 Nm). Low gains = compliance.
- **Actuator model:** custom `IdentifiedActuator` (DCMotor torque-speed curve + friction `Fs·tanh(v/0.1) + Fd·v`).
- **Rates:** physics 200 Hz (dt 5 ms), decimation 4 → policy 50 Hz. Episode 20 s.
- **Network (flat, deployed):** actor/critic MLP [128,128,128], ELU (rough config: [512,256,128]).
- **PPO:** 24 steps/env, clip 0.2, entropy 0.005, 5 epochs, 4 minibatches, lr 1e-3 adaptive (KL target 0.01), γ 0.99, λ 0.95, grad-norm 1.0, 15k iterations.
- **Key rewards:**
  - lin-vel tracking +3.0, yaw tracking +1.0
  - **biped feet-air-time +6.5** (0.3–1.0 s window), feet slide −0.35, **feet distance −0.5 (min 0.12 m)**
  - flat orientation −5.0, action rate −0.01, torque −2.5e-5
  - joint deviation: hip yaw −0.5, ankle roll −1.0, knee −0.05
  - undesired leg contacts −1.0
- **DR (on):** friction 0.2–1.25, restitution 0–0.1, link mass ×0.9–1.1, base mass −0.4/+0.3 kg, armature ×1.0–1.05, joint friction ×0.95–1.05, default pose ±0.01 rad.
- **Terminations:** timeout, base contact > 1 N.
- **Curriculum:** widens lateral command range once tracking > 80% of max.
- **Export:** `play.py` → `policy.pt` (TorchScript) + `policy.onnx`.
- **Sim2sim:** standalone Isaac Sim world. NumPy rebuild of the 44-dim obs with joint-order remap, PD gains loaded from training `env.yaml`, targets clipped to joint limits.

### 1.4 Hardware facts (`Project-Step-Up`)
- **Motors:** AK10-9 (knee, hip pitch), AK70-9 (hip roll), AK70-10 (hip yaw), AK60-6 (ankles).
- **MIT mode:** frame packs p_des, v_des, kp, kd, τ_ff with `float_to_uint` (kp clamped 0–500, kd 0–5). The motor computes τ = kp(p_des−p) + kd(v_des−v) + τ_ff itself. Hip yaw uses 11-bit standard IDs; the others use 29-bit extended IDs.
- **Threading:**
  - The ROS callback only writes `std::atomic` targets and bumps a sequence counter (acquire/relaxed ordering).
  - One `std::thread` per bus runs a `steady_clock` + `sleep_until` absolute-deadline loop (no drift) and counts overruns over 1000 µs.
  - Every 5th cycle it sends the latest target, keeping the motors fed even if the policy hiccups.
- **Ankle parallel linkage:** no clean closed-form IK. At startup, FK is swept over pitch ±35° × roll ±9° at 0.5° → LUT; IK = table search. That keeps solver iterations out of the RT loop.
- **Safety:** per-motor position clamps, MIT gain clamps, temperature/error-code checks, enable service (disabled = zero-gain MIT frames → limp), zero service, joystick-driven services.
- **Policy node:** 3 `multiprocessing` processes to get around the GIL; `Pipe` drained to the newest obs with a staleness warning; steady 20 ms timer; publishes measured loop rate; runtime picked by file extension (TorchScript or ONNX Runtime).

### 1.5 The bugs you solved (STAR material)
1. **"Actions too high":**
   - Cause: the RSL-RL actor output is **linear and unbounded** (no tanh). Combined with no obs normalizer and no action clipping, out-of-distribution inputs produced extreme joint targets.
   - Fix: added action statistics (training + play), `analyze_actions.py` for checkpoint weights, a `--clip-actions` flag, and joint-limit clipping in sim2sim.
   - Next options: lower the action scale to 0.25–0.3, or turn on empirical normalization.
   - (Note: one repo doc claims RSL-RL applies tanh. It doesn't.)
2. **Sim-to-sim consistency:** joint-order remap, gains read from `env.yaml` rather than retyped, normalizer test.
3. **Ankle parallel linkage:** turned a nonlinear IK problem into a precomputed LUT for deterministic timing.
4. **Startup ordering:** a layered launch so the IMU and CAN are live before the policy can be enabled.

### 1.6 "Why only standing balance on hardware?" (have this ready, calm and specific)
Likely contributors, in order:
- **Actuator gap:** sim PD gains/friction vs real AK motors (backlash, torque bandwidth, the MIT-mode PD running on the motor).
- **Observation gap:** base linear velocity is in the obs but has no direct sensor on hardware (say how you estimated it); IMU latency.
- **Latency:** obs → Python → ROS → CAN at 200 Hz adds delay that sim didn't model.
- **Robustness:** gain randomization and pushes were off, so the policy never saw those perturbations.

What I'd do next:
1. Measure end-to-end loop latency and add it (randomized) to sim.
2. Turn on Kp/Kd randomization and pushes.
3. Remove base lin-vel from the obs, or estimate it with a state estimator.
4. Train an **actuator network** from logged motor data (`sim2sim/utils/actuator_network.py` already exists).
5. Move inference into the C++ process (or TensorRT via the C++ API) to remove Python/IPC jitter.
6. Add real-time scheduling: `SCHED_FIFO` (prio ~80) for CAN threads, `isolcpus` + pinning, `mlockall`, a PREEMPT_RT kernel.
7. Add a **command watchdog** (hold/damp if `/tethlan/actions` goes stale) and a hardware e-stop.

### 1.7 Likely questions → short answers
- **Why one thread per bus?** Single writer per SocketCAN interface → no mutex, no write races; buses run in parallel.
- **Torn reads?** Each field is atomic, but the 6 targets aren't one snapshot, so a cycle can mix old and new for at most 5 ms. Fine at 200 Hz; a seqlock/double buffer fixes it properly.
- **Why exclude ankle velocities?** Noisiest signals with the largest sim-real error; the policy leans on positions + IMU instead.
- **Why 50 Hz over 200 Hz?** Stable PD integration at 200 Hz; 50 Hz is plenty for gait and trivial for an Orin Nano (a 128×3 MLP runs in microseconds). The bottleneck is I/O latency, not compute.
- **Why not TensorRT?** For a tiny MLP, kernel launch and IPC dominate, so TRT gains little. It matters once you add perception (height map, depth). Then: ONNX → FP16 engine, C++ runtime, validate action error vs PyTorch within tolerance.
- **How do actions become torques?** default + 0.5·a → MIT frame with stored kp/kd → motor-side PD → torque limited by the torque-speed curve.

---

## 2. Magick Mind: AI architecture lead (leadership + edge-vs-cloud + embodied agents)

### 2.1 One-minute version
> "I lead AI architecture and co-lead about 10 engineers on a 17-service platform in Go, Python and Rust: memory,
> persona, inference and voice. My main work is the Rust voice server for our ESP32 robots, the tool-calling engine
> in our open-source Rust agent SDK Mindroid, and an LLM-to-robot adapter whose principle is 'the model picks the verb,
> the robot owns the motion.'"

### 2.2 Architecture
- **Bifrost** (Go): REST gateway; gRPC to six services: Artifact, ChatHistory, Corpus, Cortex (inference), Episodic, Persona.
- **Memory:** Semantic Memory (Python, Ray actors, vector + graph); Episodic Memory keyed by (observer, observed).
- **Agents:** Spawner (Rust) runs Mindroid agents and holds `dual_brain.rs`.
- **Plumbing:**
  - etcd discovery (no hardcoded URLs)
  - Centrifugo for realtime (pub/sub + bidirectional gRPC proxy)
  - LiteLLM gateway (every key has a budget)
  - Keycloak auth
  - Docker per service → Kubernetes namespaces, Helm
- **Consolidation you drove:** 4 persona services → 1; Mindspace/End-User folded into Bifrost. Proto/`.api` files are the source of truth; a generated service graph with a `just verify` gate; design docs before code; POC promotion rules (rewrite to conventions, don't just move).

### 2.3 Voice server (Rust, Tokio)
- **Flow:** ESP32-S3 → PCM over WebSocket → Centrifugo → bidi gRPC stream per `voice:{uuid}` → voice server. (Centrifugo dials the server: it already handles auth, reconnects and fan-out.)
- **Backends:**
  - **Gemini Live** (default): local Silero VAD opens a session only on speech, with ~1 s pre-roll replay; idle timeout; new session seeded with the last 20 messages.
  - **Cascade:** VAD → Cartesia STT → LLM (LiteLLM streaming) → Cartesia Sonic TTS.
  - **Relay mode:** Gemini is only ears and mouth; a Mindroid agent decides what to say and calls tools the device advertises.
- **Concurrency:** CPU-bound Silero ONNX runs in `spawn_blocking` (one blocking thread per session) so it never starves the async runtime. A semaphore caps sessions at 50, lowered to vendor quotas (Gemini 4, Ink 2: 8).
- **Barge-in:**
  - 200 ms debounce, fires once per turn, only while the agent is speaking (echo gate).
  - With Ink 2 it waits for an interim transcript, so "uh-huh" doesn't cut the agent off.
  - Cancel: pipeline token + Cartesia context + `CONTROL_BARGE_IN` so the device flushes its speaker.
- **First-audio latency:**
  - `TextBatcher` flushes at 20/60/120 chars or a sentence boundary; one TTS continuation context per turn (no per-sentence handshake).
  - Telemetry tracks `first_audio_ms`.
  - Biggest remaining cost: batch STT adds 200–1500 ms after silence, so semantic endpointing (Ink 2) is the fix.
- **Edge relevance:** this is a real-time streaming pipeline with a latency budget, backpressure, and CPU vs async isolation. Same thinking as perception pipelines on a robot.

### 2.4 Embodied agents and safety (`pocs/AGD_MM_FC_ROBOT`)
- Two processes with a typed contract (`proto/robot.proto`): a Rust Mindroid node (capabilities as tools) and a Python robot node (MuJoCo + SO-ARM101 hardware). Zenoh on the `poc/zenoh` branch.
- The robot publishes a **capability manifest** (`interruptible`, `blocks_base`, `safety_json` like `max_speed_mps 0.8`). The LLM picks a capability **name**; the robot **ignores LLM-supplied params**. Trajectories, speed caps and joint clamps (`FC_ARM_MAX_REL`) are deterministic code.
- **Preemption:** `robot/{id}/cancel` with SAFETY/TIMEOUT reasons → back to home; Ctrl-C disables torque; power switch is the final e-stop.
- **Devices hold zero API keys:** token broker issues 30-min HMAC tokens per device.
- **Honest scope:** robot-side latency hasn't been measured, and the voice→arm loop is designed, not yet built.

### 2.5 Fast/smart inference (dual brain)
- Each turn is triaged as silent / reflex (canned) / fast (cheap LLM) / smart (capable LLM via Cortex SmartBrain).
- A POC replaces LLM triage with a typed classifier verdict: ~100–140 ms of a ~1.6 s turn. Low confidence escalates (reflex needs ≥ 0.8).
- **Maps directly onto robots:** a reflex/fast path onboard, slow deliberation offboard. That is the System-1/System-2 split in humanoid VLA stacks.

### 2.6 Likely questions → short answers
- **Edge vs cloud, how did you decide?**
  - On the device: latency, privacy and safety work (wakeword, codec, motion, safety).
  - In the cloud: heavy reasoning, behind a token broker.
  - Mindroid feature flags (`llm-local` Ollama, `persistence` SQLite, `transport-audio`) build the same agent for edge or cloud.
  - On a humanoid with a Jetson, I'd move VAD + STT + a small LLM onboard and keep the smart model offboard.
- **How do you make LLM-driven robots safe?** Bounded verbs from a manifest, deterministic motion, numeric limits in the contract (not in the prompt), preemption, hardware e-stop.
- **How do you lead architecture across 10 engineers?** Contract-first (protos), design docs, generated service graph + CI gate, POC promotion rules, consolidating services when they multiply.
- **Disagreement or trade-off example:** capping sessions to vendor quotas vs the 50 design target, or consolidating 4 persona services into 1.

---

## 3. Raggity RAG (evaluation discipline + classic vs neural judgment)
- **What:** FastAPI RAG service.
  - Recursive chunking (400 chars / 120 overlap in eval).
  - Pluggable embedders: hash baseline, bge-small 384-d, OpenAI, Ollama.
  - SQLite: float32 brute-force dot product + **FTS5 BM25**; hybrid via **RRF k=60** (30 candidates each).
  - Optional cross-encoder rerank (MiniLM-L6); grounded answers with numbered citations and refusal.
  - Fully local path (Ollama); Prometheus per-stage timings; Docker.
- **Eval:** 35 questions / 23 docs, document-level labels (survive re-chunking), tagged exact-term / paraphrase / multi-doc.

| config | hit@1 | MRR | p50 |
|---|---|---|---|
| hash dense | 0.37 | 0.52 | 0.1 ms |
| BM25 | 0.69 | 0.81 | 0.2 ms |
| **bge-small dense** | **0.97** | **0.99** | 5.9 ms |
| hybrid (RRF) | 0.94 | 0.97 | 6.3 ms |
| dense + rerank | 0.97 | 0.99 | 43 ms |
| BM25 + rerank | 0.97 | 0.99 | 27.8 ms |

- **Talking points:**
  - Profiling showed `embed_query` = 5.8 of 6.2 ms, so the model is the bottleneck. On edge, quantize or distill the embedder, not the index.
  - Hybrid lost because short keyword-dense docs pollute RRF.
  - The reranker costs 6–9× latency for zero gain over dense, but rescues BM25 → **BM25 + reranker is the cheapest good stack with no embedder.**
  - Honest caveat: small, saturated, author-biased eval → treat it as an upper bound.
- **Use it for:** "classic vs neural" and "how do you evaluate" questions. Robot link (on-device knowledge store) is a **proposal**, not built.

---

## 4. edge-llm-lab (in progress; no numbers to quote)
- **Question it answers:** "With N GB and context C, which model + quantization, how much quality lost, how fast?"
- **Design:**
  - llama.cpp CUDA on **Orin Nano 8 GB** (68 GB/s, 102 GB/s in MAXN SUPER on JetPack 6.2).
  - 2 instruct models (0.5–1.5B) × GGUF Q4 / Q8 / FP16 × prompts of 128 / 512 / 1024 tokens; 5 repeats; cold load separate from warm runs.
  - 4 GiB app memory ceiling.
  - Metrics: TTFT, prefill & decode tok/s, **decode as % of bandwidth roofline**, peak memory, J/token, JSON-extraction F1 + invalid-JSON rate, KL drift vs FP16.
  - Logs power mode/clocks/temperature (tegrastats); schema-validated, append-only results; pinned commits/hashes.
- **Concepts to explain fluently:**
  - **Decode is memory-bandwidth bound:** each token streams all weights → ceiling ≈ bandwidth / model bytes. 1.5B @ Q4 ≈ 0.9 GB → ~70–75 tok/s max at 68 GB/s.
  - Prefill is compute bound (drives TTFT).
  - **KV cache** = layers × 2 × kv_heads × head_dim × ctx × bytes. It grows linearly and can eat the memory quantization saved.
  - Unified memory: OS + CUDA context share the 8 GB.
  - Power mode is a controlled variable.
- **If you can, run part of the v0.1 matrix on your Orin Nano before Thursday.** Even one model × three formats turns this into real evidence for the JD's quantization/memory bullets.

---

## 5. Study material for the TensorRT / quantization / pruning gap
(Lessons from the graduation project in `Project-Step-Up/Deep-Learning-Models-Optimization...`. These are **not your work**; use them as known results.)
- **Pipeline:** PyTorch → ONNX → `trtexec --onnx=m.onnx --saveEngine=m.trt --fp16` (or `--int8` + calibration cache) → Python/C++ runtime.
- **Unstructured pruning gave no speedup** on Jetson: dense kernels ignore zeros. **Structured pruning** (channel removal via Torch-Pruning dependency graphs, NVIDIA ModelOpt FastNAS) gave real latency gains, followed by fine-tuning.
- **Quantization ladder:** FP16 (nearly free on Ampere/Volta) → PTQ INT8 static with calibration set → QAT when PTQ drops too much accuracy. Dynamic-range INT8 = weights only.
- **Reported:** MobileNetV2 9.88 MB → 0.47 MB, 31.9 → 66.1 FPS with TRT FP16 on Jetson Nano; VGG11 ~16× faster for ~2% accuracy.
- **Distillation** helped little for an already-small MobileNet. Low-rank SVD is another option for FC-heavy models.
- **How to bring it to your own experience:** "My policies were small MLPs where TensorRT gains little, so I used ONNX Runtime/TorchScript. For perception models I'd follow this pipeline: profile → FP16 engine → INT8 with a representative calibration set and per-layer sensitivity checks → structured pruning + fine-tune if still over budget → validate on target at p99 latency."

---

## 6. Map your projects onto the JD (use these bridges in answers)

| JD bullet | Your evidence |
|---|---|
| Own the end-to-end ML stack | Tet Hlan: reward design → PPO → export → Jetson inference → C++ motor stack |
| Accuracy / latency / power, edge vs cloud | Dual-brain fast/smart routing; voice: device vs cloud split; Raggity latency-vs-quality table |
| Jetson deployment | Orin Nano 50 Hz policy; edge-llm-lab design (bandwidth roofline, unified memory) |
| Quantization / TensorRT / pruning | ⚠️ Weakest. Section 5 + edge-llm-lab design; be honest and show you know the method |
| Traditional ML vs DL | BM25 + reranker vs dense embeddings; LUT kinematics vs a solver; PD control under a learned policy |
| Embedded Linux, multi-process, containers | Multi-process policy node, SocketCAN, ROS 2 launch ordering, Docker/K8s at Magick Mind |
| CPU isolation, thermal | ❌ Not done. Explain the plan (section 1.6, item 6) |
| C++ and Python in production | C++ RT CAN stack + Python policy node; Rust/Go/Python services |
| Robotics fundamentals | MIT-mode PD, parallel-linkage kinematics, IMU, ROS 2, sim2real, MuJoCo |
| Leadership, design reviews | Tet Hlan team lead; ~10-engineer co-lead, contract-first design, POC promotion rules |
