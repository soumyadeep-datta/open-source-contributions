# Open Source Contributions

A log of my contributions to open-source projects, focused on AI/ML systems, LLM inference and serving, GPU/CUDA, and distributed systems.

---

## Implementing GGML_OP_TOP_K for the CUDA backend — llama.cpp

**Project:** [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
**Issue:** https://github.com/ggml-org/llama.cpp/issues/14909
**Status:** In Progress

### Why I Chose This Issue

TOP_K returns the indices of the k largest elements per row — a core primitive in LLM
inference, used in top-k sampling and ranking. It's implemented for the CPU backend in
ggml (llama.cpp's tensor library) but is currently unsupported on CUDA (marked ❌ in
docs/ops.md), so on a GPU it can't run natively. I'm implementing the CUDA version.

I chose it because it sits squarely in my focus area of high-performance ML systems and
GPU programming, and it builds directly on prior CUDA work I've done implementing GPU
top-k retrieval for vector search. On the GPU, TOP_K is a genuine parallel-selection
problem — warp-level reductions and partial selection rather than a trivial elementwise
map — which makes it a substantive systems contribution. The scope is well-bounded (one
operation, one PR), and the definition of done is crisp: it must pass
`test-backend-ops -o TOP_K` against the trusted CPU reference implementation, with
performance reported via `test-backend-ops perf -o TOP_K`.
