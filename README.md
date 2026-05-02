<div align="center">
  <img src="assets/dashboard-top.png" alt="LLM Inference Bench Dashboard" width="100%" style="border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.4);">

  <br>
  <h1>⚡ LLM Inference Bench — Apple Silicon</h1>
  <p><b>An empirical approach to finding the sweet spot between Speed, RAM, and Quality on Mac</b></p>

  <p>
    <a href="https://github.com/saciducak/LLM-Inference-Bench-Mac/issues"><img src="https://img.shields.io/badge/Status-Active-success.svg?style=flat-square" alt="Status"></a>
    <img src="https://img.shields.io/badge/Platform-Apple_Silicon-000000.svg?style=flat-square&logo=apple" alt="Platform">
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=flat-square&logo=python&logoColor=white" alt="Python">
  </p>
</div>

---

## 👋 The Backstory: Why I Built This

As an AI engineer working with local models on Apple Silicon, I realized I was spending too much time guessing. *"Should I use MLX or llama.cpp for this project? Is Q4_K_M really better than Q8_0 in terms of TTFT (Time to First Token)? How much does Turkish generation quality degrade at 4-bit?"*

Instead of relying on Reddit threads or theoretical FLOPS, I decided to build a **data-driven benchmarking suite**. This project is a reflection of my engineering philosophy: **build the tools you need to measure what matters.** 

It systematically tests the same prompts across different runtimes (Ollama, llama.cpp, MLX) and quantization levels, tracks the actual hardware metrics in the background, and visualizes the trade-offs on a premium local dashboard.

---

## 📊 Visualizing the Trade-offs

Optimization is never a straight line. By tracking multiple metrics simultaneously, I was able to map out the exact Pareto frontiers for my M1 Pro machine.

<div align="center">
  <img src="assets/dashboard-charts.png" alt="Dashboard Charts" width="90%" style="border-radius: 8px;">
  <p><i>Left: Peak RAM usage across quantizations. Right: Quality vs. Speed (Bubble size = RAM footprint).</i></p>
</div>

### Key Learnings During Development:
1. **Apple's Unified Memory (UMA) is a beast:** By using MLX (`mlx-lm`), we achieve zero-copy memory access, drastically reducing load times and TTFT compared to traditional REST API overheads.
2. **The Quantization "Sweet Spot":** For Turkish text generation, dropping down to Q2_K saves a ton of RAM but severely hurts coherence. 4-bit (MLX) and Q4_K_M (llama.cpp) consistently provided the best balance of Tokens/sec and Quality.
3. **Quality Evaluation is Hard:** Speed doesn't matter if the model hallucinates. I had to build a custom deterministic NLP evaluator that specifically scores Turkish responses based on length, keywords, semantic coherence, and character density.

---

## 📋 The Detailed Results View

The suite doesn't just output terminal logs; it generates a structured JSON that feeds into a custom-built, glassmorphism UI for deep analysis.

<div align="center">
  <img src="assets/dashboard-results.png" alt="Detailed Results Table" width="90%" style="border-radius: 8px;">
</div>

---

## 🏗️ How It Works Under the Hood

I designed the architecture to be modular so I can easily plug in new backends as the ecosystem evolves.

* **Abstract Base Backend:** An `InferenceBackend` class ensures uniform metric collection whether we are hitting an Ollama REST API, running `llama-cpp-python` with Metal offload, or executing native `mlx`.
* **Asynchronous Telemetry:** A `MemoryTracker` context manager spins up a background thread using `psutil` to sample Resident Set Size (RSS) peak memory during the exact inference window.
* **Warmup Cycles:** The orchestrator enforces strict warmup runs to separate cold-start model I/O times from actual decode throughput.
* **Vanilla JS Dashboard:** I wanted a beautiful UI without the overhead of React/Next.js for a simple local tool. The dashboard uses Chart.js, CSS variables, and pure JavaScript.

---

## ⚡ Quick Start

Want to see how your Mac handles it?

### 1. Setup
```bash
git clone https://github.com/saciducak/LLM-Inference-Bench-Mac.git
cd LLM-Inference-Bench-Mac

# Install core dependencies
pip install -r requirements.txt

# Install runtimes (optional but recommended)
pip install mlx-lm
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```

### 2. Run the Benchmark
```bash
# Fast sanity check (1 prompt, 1 iteration)
python run_benchmark.py --quick --runtime ollama

# Full evaluation suite
python run_benchmark.py
```

### 3. Launch the Dashboard
```bash
python run_benchmark.py --dashboard
```

---

## 🚀 What's Next?

This project was a fantastic deep dive into hardware-aware ML engineering. Moving forward, I plan to:
- Integrate **vLLM** (once Apple Silicon support matures further).
- Add energy consumption metrics (tracking Watts during generation).
- Implement an automated LLM-as-a-judge for even more robust quality scoring.

*If you're an AI engineer optimizing edge deployments, feel free to fork this or open a PR. Let's build better benchmarks!*
