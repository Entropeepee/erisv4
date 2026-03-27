# Deep Research Prompt — Eris Echo v4 Technology Stack Audit (March 2026)

## Instructions for Gemini

You are conducting a comprehensive technology audit for a GPU-accelerated cognitive architecture being built on a Windows desktop with an NVIDIA RTX 5080 (16GB VRAM, Blackwell sm_120 architecture). The system uses **CuPy** (NOT PyTorch — PyTorch does not support sm_120 Blackwell GPUs as of March 2026). The developer has limited coding experience and needs clear, actionable recommendations.

## Research Tasks

### 1. CuPy on RTX 5080 Blackwell (CRITICAL)

- What is the **latest stable CuPy version** as of March 2026 that fully supports RTX 5080 / sm_120 / Blackwell?
- What CUDA toolkit version is required? Is `cupy-cuda12x` still the correct pip package, or has a newer variant been released?
- Are there any known CuPy bugs or performance regressions on Blackwell (sm_120) as of March 2026?
- What is the status of `cupyx.scipy.ndimage` on CUDA 13.x? (Previously broken — has this been fixed?)
- Best practices for VRAM management with CuPy: memory pool configuration, pinned memory, async transfers.
- Are CuPy RawKernels and ReductionKernels fully functional on sm_120?
- Has CuPy added any new features since v13 that are relevant to PDE solvers or stencil operations?

### 2. GPU-Accelerated PDE Solvers (State of the Art)

- What are the best GPU PDE solver libraries compatible with CuPy as of March 2026? Consider: cuFFT-based spectral methods, finite difference stencils, reaction-diffusion systems.
- Is there a CuPy-native way to do 2D convolution / stencil operations that is faster than `cp.roll`-based manual stencils?
- Are there any new CuPy integrations with NVIDIA's cuSPARSE, cuSOLVER, or cuTENSOR that would accelerate Laplacian computation on a 64×64 or 128×128 grid?
- What is the current state of the art for GPU-accelerated reaction-diffusion PDE systems? Any libraries or papers from 2025-2026 worth knowing about?

### 3. Hex Lattice / Graph Computation on GPU

- Best approaches for hexagonal grid computation on GPU with CuPy (not PyTorch). Options: structured arrays with offset coordinates, sparse adjacency matrices, custom CUDA kernels.
- Is there a CuPy-compatible graph neural network or graph computation library that handles non-rectangular lattices?
- Performance comparison: axial coordinate lookup tables vs. sparse matrix multiplication for hex neighbor propagation.

### 4. FAISS + CuPy Integration

- Current state of FAISS GPU support on RTX 5080 / Blackwell. Does `faiss-gpu` work on sm_120?
- If FAISS GPU doesn't support Blackwell yet, what are the best alternatives for GPU-accelerated vector similarity search with CuPy? Consider: cuVS (NVIDIA's new vector search library), custom brute-force with CuPy, Annoy, ScaNN.
- Memory-mapped index files for long-term memory with millions of entries: best practices.

### 5. FastAPI + Async + GPU

- Best practices for building a FastAPI server that calls GPU-accelerated CuPy code. Threading model, async patterns, avoiding GIL issues.
- Should CuPy GPU operations run in a separate process (multiprocessing) or can they coexist with FastAPI's async event loop?
- Any new async GPU libraries or patterns from 2025-2026?

### 6. Embedding Models for Semantic Memory

- What are the best local embedding models as of March 2026 that run on GPU WITHOUT PyTorch? Options: ONNX Runtime with CUDA, TensorRT, CuPy-based inference.
- Can `sentence-transformers` or equivalent run through ONNX Runtime on Blackwell?
- Recommended embedding model for a system that needs to embed text chunks for FAISS-based semantic retrieval. Model size should fit alongside the main application in 16GB VRAM (allocate at most 2GB for the embedding model).

### 7. Cloud LLM API Race Pattern

- Best practices for racing multiple LLM API calls (OpenAI, Google Gemini, Cerebras) in Python as of March 2026. Use `asyncio.gather` with `return_when=FIRST_COMPLETED`? `httpx` vs `aiohttp`?
- Any new Python LLM client libraries that support unified multi-provider racing?
- Rate limiting and cost management patterns for multi-provider setups.

### 8. Windows-Specific Considerations

- Any Windows-specific gotchas for CuPy, FAISS, or FastAPI as of March 2026?
- Best Python version for this stack on Windows (3.11 vs 3.12 vs 3.13)?
- Conda vs pip vs uv for dependency management — what's the current recommendation for CUDA-heavy Windows projects?

### 9. Checkpoint / Resume Patterns

- Best practices for checkpointing and resuming long-running GPU computations in Python. Consider: pickle vs. safetensors vs. numpy `.npz` vs. CuPy `.npy` for saving GPU array state.
- How to implement a robust checkpoint-every-N-minutes pattern that survives crashes (ADHD interruptions, pets, toddlers, power outages).

### 10. Testing GPU Code

- How to test CuPy GPU code on machines that may or may not have a GPU. Mocking strategies, CPU fallback patterns.
- Any CuPy-specific testing frameworks or utilities as of March 2026?

## Output Format

For each section, provide:
1. **Current best practice** (what to use right now, March 2026)
2. **Specific versions** (exact package names and version numbers)
3. **Known issues** (bugs, incompatibilities, gotchas)
4. **Code snippet** (minimal working example where applicable)
5. **Sources** (links to documentation, release notes, or papers)

Prioritize actionable, tested information over theoretical possibilities. If something is experimental or unstable, say so explicitly.
