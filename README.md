# CUDA Rasterizer

**Hangyu Zhang**

A high-performance GPU rasterizer implemented entirely in CUDA, simulating modern hardware pipeline and features (Nvidia's tiled caching rendering pipeline)

---

*Stanford Bunny rendered at 1920x1080 with texture mapping and trilinear filtering*

*upper - opaque; bottom - OIT*

![Rendered Bunny Model](./images/teaser.png)

![OIT](.\images\OIT.png)

### Motivation - Why Build a CUDA Rasterizer?

Traditional graphics APIs (OpenGL, DirectX, Vulkan) abstract away the rendering pipeline. This project peels back those layers to implement every stage manually on the GPU, providing deep insight into:
- How modern GPUs organize rendering work into tiles for better cache efficiency
- Memory access patterns and atomics for lock-free data structures
- Parallel primitive processing and stream compaction
- CUDA-OpenGL interop for zero-copy buffer sharing
- Performance optimization through CUDA Graphs

## Pipeline Architecture

### Work flow

![Workflow](./images/workflow.png)

| Stage | Kernel | Description |
|-------|--------|-------------|
| **1** | `VertexFetchAndShading` | Transform vertices to clip space (MVP matrix) |
| **2** | `PrimitiveAssembly` | Generate triangles, frustum clipping, backface culling |
| **3** | `PrimitiveCompaction` | Stream compact to remove culled primitives |
| **4** | `TriangleSetup` | Compute edge equations, depth equations, AABBs, interpolation coefficients |
| **5** | `PrimitiveBinning` | Assign triangles to 128x128 screen bins |
| **6** | `CoarseRasterizer` | Subdivide bins into 8x8 pixel tiles & compute tile cover and hierarchical z |
| **7** | `FineRasterizerWIP` | Generate fragments in 2x2 quads for derivatives & compute pixel coverage for fragment in quad and early-z |
| **8** | `PixelShader` | per fragment shading |
| **9** | `ROP` | Depth test, blending, write to render target (preserve API order) |

## Key Features

### Modern Tile-based Rendering Architecture

We implemented a sort-middle rendering architecture (the primitives are redistributed by binning), which is similar to Nvidia's Tile caching immediate mode rendering(TCR) after Maxwell architecture and AMD's Draw stream binning rasterization (DSBR).

(Since we cannot get the real implementation of both their architectures, I have to make some self experiment to guess their structure. We introduced a binning process and assigns each GPC(block) with a big tile on the screen. I believe it's more like DSBR - in contrast TCR assigns GPC with a set of small tiles in one big tile with a checkboard pattern)

- **Two-Level Tiling System beyond Fine Raster**: Screen divided into 128x128 pixel bins, a chunk of triangles will first test coverage with each bin on the screen. Further the bins are subdivided into 8x8 pixel tiles, testing triangle-tile intersection before queuing work for fine rasterization.. This system helps us balance the workload among different SMs and reduce the amount of fine rasterization.
- **Complicated Z-test System**: We use a hi-z to test if a triangle is fully occluded by other in a coarse level and early-z to test if a fragment should pass into the PS stage. This strategy further reduces the overhead of fine rasterization and the amount of overdraw for pixel shader.
- **Quad-based Pixel shader**: In fine rasterization, we output each fragment in a quad form (with 3 adjacent fragments). And parallelly process 8 quads in one warp for PS.
- **Restrict API Order and OIT**: In order to make fragment write order obey primitive sequence, we do a caching and sorting for all fragments write to each pixel. The sort based on primitive ID and be replaced with depth in order to implemented a OIT. 

![zoom in and out](./images/zoom_in.gif)

As the figure above shows, because of the task reducing and redistributing system of the tile-based arch, it's smooth to both zoom in and out

### Advanced CUDA Primitives & technique []

* Use **warp level functions** implemented a high performance block Size scan algorithm
* Use **cooperative groups** implemented a one pass stream paction which allows compaction up to **500,000** primitives
* Efficient usage of shared mem and atomic operation
  * reduce atomic allocation operations by one block/warp issue one (minimize stall counts)
  * change per thread atomic access to shared mem (minimize stall cycles) 
* Use **persistent threading** to reduce thread **divergence** and **unbalance** work load in a block

### Advanced Texturing

- **Trilinear Filtering** with automatic LOD calculation (based on quad info)
- CPU-side mipmap generation
- **DDX/DDY derivative** computation using warp shuffle operations
- an optimized **tile linear storage** structure for texture

![bilinear](.\images\sampling_bilinear.png) ![trilinear](.\images\sampling_trilinear.png)

left side: sampling with bilinear interpolation, right side: sampling with trilinear interpolation

with trilinear interpolation, we get a less noisy texture (blurred) which dramatically increase rendering quality for objects at distance

### CUDA Multi-Stream & Graph Optimization []

- use cudaStream & cudaGraph to capture a "pipeline object" for the rasterizer and reuse it every frame
- reducing overhead of launch small kernels in the pipeline 
- leverage the parallel capability of different GPU tasks (copy & kernel) and between CPU and GPU (async copy & launch)

## Optimization & Performance Analysis

### Test Scene
- **Model**: Stanford Bunny (5000 triangles) Cow (5000 triangles)

  **As the cooperative group's limitation on launched kernel block size, currently we cannot process model have more than 10k triangles. So we test our pipeline with model about 5000 triangles

- **Resolution**: 1920x1080

### Baseline Performance (tiled base method without optimizing)

![before](./images/origin.png)

As shown in the figure, the goal of optimization should focus on kernels "CoarseRasterizer", "FineRasterizer", "PixelShader", "ROP", and "StreamingToFrameBuffer" as they take much of the frame time.

<details>   <summary>Optimization: CoarseRasterizer</summary>      As it has a small kernel size (a block for one bin), reducing register usage here makes no sense. Hence we increase the loop unrolling count, increasing instruction effectiveness.   </details>

<details> <summary>Optimization: FineRasterizer</summary>We rewrite the queue read logic and fragment write back logic, making global read and write amortized inside the warp, reducing warp divergence. Interestingly, by making every 4 threads write a quad back, we increase memory coalescing and reduce the register overhead of loop unrolling, which also increases occupancy. Besides that, we increase our block size from 32 to 256, let every 32 threads (a warp) process one tile and the whole block process 4 tiles simultaneously. This dramatically increases occupancy, hiding read latency and eliminating the tail effect.</details>

<details> <summary>Optimization: FineRasterizer</summary> We rewrite the queue read logic and fragment write back logic, making global read and write amortized among the warp, reducing warp divergence. Interestingly, by making every 4 threads write a quad back, we increase memory coalescing and reduce the register overhead of loop unrolling, which also increases occupancy. Besides that, we increase our block size from 32 to 256, let every 32 threads (a warp) process one tile and the whole block process 4 tiles simultaneously. This dramatically increases occupancy, hiding read latency and eliminating the tail effect.
</details>
<details> 
    <summary>Optimization: PixelShader
    </summary>
    We optimize the fragment data structure, allowing each thread of the warp to read and write continuous 16-byte (float4) data in global mem. Secondly, as kernel arguments have to load to register from constant cache, when the parameter is large, the register may spill into local memory, which introduces memory stalls when accessing. We optimize the Texture2D structure and method to read from texture.
    <br />
    <img src="F:\CIS5650\CudaRasterizer\images\optimized_Frag.png" style="zoom:33%;" />
    <img src="F:\CIS5650\CudaRasterizer\images\optimized_tex.png" style="zoom:25%;" />
    <br />
    The figure above shows our optimized data structure, see source code for more details

<details> 
    <summary>Optimization: ROP Stage</summary>
    For the same reason, we packed the structure for FragmentOut. Besides that, we use shared mem to eliminate local memory usage and reduce register usage, which makes the kernel's ideal occupancy reach 100%. <mark>But, unfortunately, increasing the occupancy lets more blocks stay alive on SM, which may lead to more eviction for cachelines, making L1 & L2 hit rate lower<\mark>. So, we do not see big improvements for this kernel. 

<details> <summary>Optimization: StreamingToFrameBuffer</summary>As it has a memory-bound nature, what we do is just making memory access coalescing.</details>

With these optimization, we get a 31.7% improvement on performance:

### Final Performance

![after](F:\CIS5650\CudaRasterizer\images\after.png)

### CUDA Graph Impact

| Metric | Standard Path | CUDA Graph | Improvement |
|--------|--------------|------------|-------------|
| CPU Overhead | 450 μs | 320 μs | **-28.9%** |
| GPU Time | 1407 μs | 1390 μs | -1.2% |
| Total Frame Time | 1857 μs | 1710 μs | **-7.9%** |

### CUDA Graph Optimization

CUDA Graphs reduce CPU overhead by capturing kernel launches into a reusable execution graph:

```cuda
// RasterizerGraph.cu - First frame captures graph
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// 12 parallel memset operations across streams
cudaMemsetAsync(dBuffer1, 0, size, stream1);
cudaMemsetAsync(dBuffer2, 0, size, stream2);
// ... etc

// Sequential rendering kernels (data dependencies)
VertexShading<<<grid, block, 0, stream>>>();
PrimitiveAssembly<<<grid, block, 0, stream>>>();
// ... rest of pipeline

cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

// Subsequent frames just launch the graph
cudaGraphLaunch(graphExec, stream);
```

**Performance Impact**:

- CPU overhead: -20-30% (measured with CPU profiler)
- Frame time: Marginal improvement (GPU-bound)
- Best use case: CPU-bound scenarios, many small kernels

**Critical Limitation**: OpenGL interop (`cudaGraphicsMapResources`) **cannot** be captured in graphs. Solution: Render to staging buffer, then `cudaMemcpy` to PBO outside graph.

### Limitation & Failure Case

**Fine Rasterization (27% of frame)** is the primary bottleneck. Optimization opportunities:
1. **Hierarchical Z-buffer**: Skip tile rasterization if fully occluded
2. **Increase tile size**: 8x8 tiles may be too small, causing overhead. Test 16x16.
3. **Warp utilization**: Profile to ensure >50% warp occupancy

**Pixel Shading (22.7%)** dominated by texture sampling. Mitigations:
- **Texture cache optimization**: Pad textures to 128-byte alignment
- **Reduce trilinear to bilinear**: 30% faster, minimal quality loss
- **Group fragments by tile**: Improve texture cache hit rate



## Future Enhancements

- [ ] **Hierarchical Z-Buffer (Hi-Z)**: Early rejection of occluded tiles
- [ ] **Multiple Render Targets (MRT)**: G-buffer for deferred shading
- [ ] **Compute Shader Pipeline**: Port to pure compute for Vulkan/DX12 comparison
- [ ] **Multi-Draw Indirect**: Batch multiple objects without CPU intervention
- [ ] **Compressed Textures**: BC7 support for reduced bandwidth
- [ ] **MSAA**: Multi-sample anti-aliasing with coverage masks
- [ ] **Dynamic Branching**: Uber-shader for multiple material types



## References

- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [A trip through the Graphics Pipeline](https://fgiesen.wordpress.com/2011/07/09/a-trip-through-the-graphics-pipeline-2011-index/)
- [Tile-Based Rendering](https://developer.arm.com/documentation/102662/0100/Tile-based-rendering)
- [CUDA Graphs Introduction](https://developer.nvidia.com/blog/cuda-graphs/)
- [GPU Gems 2: Stream Compaction](https://developer.nvidia.com/gpugems/gpugems2/part-vi-simulation-and-numerical-algorithms/chapter-39-parallel-prefix-sum-scan)
