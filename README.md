# CUDA Rasterizer

**Hangyu Zhang**

A high-performance GPU rasterizer implemented entirely in CUDA, simulating modern hardware pipeline and features (Nvidia's tiled caching rendering pipeline)

---

*Stanford Bunny rendered at 1920x1080 with texture mapping and trilinear filtering*

*Top - opaque; Bottom - OIT (just object level)*

<img src="./images/teaser.png" style="zoom:25%;" />

<img src="./images/OIT.png" style="zoom:25%;" />

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
| **7** | `FineRasterizer` | Generate fragments in 2x2 quads for derivatives & compute pixel coverage for fragments in quad and do early-z |
| **8** | `PixelShader` | Per-fragment shading |
| **9** | `ROP` | Depth test, blending, write to render target (preserve API order) |

## Key Features

### Modern Tile-based Rendering Architecture

We implemented a sort-middle rendering architecture (the primitives are redistributed by binning), which is similar to Nvidia's Tile caching immediate mode rendering(TCR) after Maxwell architecture and AMD's Draw stream binning rasterization (DSBR).

 We introduce a binning process and assign each GPC(block) with a large tile on the screen. I believe it's more similar to DSBR - in contrast,  TCR assigns GPCs with a set of small tiles in one large tile with a checkerboard pattern. (Since we cannot get the real implementation of either architecture, we had to conduct experiments to infer what they have done (see reference).)

- **Two-Level Tiling System beyond Fine Raster**: The Screen is divided into 128x128 pixel bins. A chunk of triangles first tests coverage with each bin. Furthermore, the bins are subdivided into 8x8 pixel tiles, testing triangle-tile intersection before queuing work for fine rasterization. This system helps us balance the workload among different SMs and reduce the computation for fine rasterization.
- **Complex Z-test System**: We use a Hi-z buffer to test if a triangle is fully occluded by others at a coarse level, and early-z to test if a fragment should pass into the PS stage. This strategy further reduces the computation for fine rasterization and the amount of overdraw for the pixel shader.
- **Quad-based Pixel Shader**: In fine rasterization, we output each fragment in a quad form (with 3 adjacent fragments) adn process 8 quads in parallel in each warp for PS.
- **Preserve API Order and OIT**: To make fragment write order obey the primitive sequence, we cache and sort all fragments written to one pixel. The sort is based on primitive ID and can be replaced with depth in order to implemented OIT. 

![zoom in and out](./images/zoom_in.gif)

As the figure above shows, because of our tile-based work redistribution rasterization system, it's smooth to both zoom in and out

### Advanced CUDA Primitives & Techniques [See Chapter Optimization & Performance Analysis]

* Based on **warp level functions**, we implemented a high performance block-size scan algorithm
* Using **cooperative groups**, we implemented a one-pass stream compaction which allows compaction of up to **500,000** primitives
* Efficient usage of shared mem and atomic operations
  * Reduce atomic allocation operations by having one block/warp issue one (minimize stall counts)
  * Make per-thread atomic access to shared mem instead of global mem (minimize stall cycles) 
* Use **persistent threading** to reduce thread **divergence** and **unbalanced** work load in a block

### Advanced Texturing

- **Trilinear Filtering** with automatic LOD calculation (based on quad info)
- CPU-side mipmap generation
- **DDX/DDY derivative** computation using warp shuffle operations
- An optimized **tile linear storage** structure for textures

![bilinear](./images/sampling_bilinear.png) ![trilinear](./images/sampling_trilinear.png)

Left side: sampling with bilinear interpolation, right side: sampling with trilinear interpolation

With trilinear interpolation, we get a less noisy texture (blurred) which dramatically increases rendering quality for objects distant

### CUDA Multi-Stream & Graph Optimization [See Chapter Optimization & Performance Analysis]

- Use cudaStream & cudaGraph to capture a "pipeline object" for the rasterizer and reuse it every frame
- Reducing the overhead of launch small kernels in the pipeline 
- Leverage the parallel capability of different GPU tasks (copy & kernel) and between CPU and GPU (async copy & launch)

## Optimization & Performance Analysis

### Test Scene
- **Model**: Stanford Bunny (5000 triangles) Cow (5000 triangles)

  **Due to the cooperative group's limitation on launched kernel block size, currently we cannot process models that have more than 10k triangles. So we test our pipeline with models of about 5000 triangles

- **Resolution**: 1920x1080

### Kernel Performance

**Baseline Performance (tile-based method without optimizing)**

![before](./images/origin.png)

As shown in the figure, the optimization target should be the kernels "CoarseRasterizer", "FineRasterizer", "PixelShader", "ROP", and "StreamingToFrameBuffer" as they take much of the frame time.

<details>   <summary>Optimization: CoarseRasterizer</summary>      As it has a small kernel size (a block for one bin), reducing register usage here makes no sense. Hence we increase the loop unrolling count, increasing instruction effectiveness.   </details>

<details><summary>Optimization: FineRasterizer</summary> 
    We rewrote the queue read logic and fragment write back logic, making global read and write amortized among the warp, reducing warp divergence. Interestingly, by making every 4 threads write a quad back, we increase memory coalescing and reduce the register overhead of loop unrolling, which also increases occupancy. Besides that, we increased our block size from 32 to 256, let every 32 threads (a warp) process one tile and the whole block process 4 tiles simultaneously. This dramatically increases occupancy, hiding read latency and eliminating the tail effect.</details>
<details> 
    <summary>Optimization: PixelShader
    </summary>
    We optimized the fragment data structure, allowing each thread of the warp to read and write continuous 16-byte (float4) data in global mem. Secondly, as kernel arguments must load to registers from the constant cache, when the parameter itself is large, the registers may spill into local memory, which introduces memory stalls when accessing. We optimized the Texture2D structure and method to read from texture.
    <br />
    <img src="./images/optimized_Frag.png" style="zoom:33%;" />
    <img src="./images/optimized_tex.png" style="zoom:25%;" />
    <br />
    The figure above shows our optimized data structure, see source code for more details.
</details>

<details> 
    <summary>Optimization: ROP Stage</summary>
    For the same reason, we packed the structure for FragmentOut. Besides that, we use shared mem to eliminate local memory usage and reduce register usage, which makes the kernel's ideal occupancy reach 100%. 
    <mark>However, unfortunately, increasing the occupancy lets more blocks stay alive on SM, which may lead to more evictions for cache lines, making L1 & L2 hit rates lower.
    </mark> 
        So, we do not see large improvements for this kernel. 
</details>

<details> 
    <summary>Optimization: StreamingToFrameBuffer</summary>As it has a memory-bound nature, what we do is simply ensure memory access is coalesced.
</details>


With these optimizations, we achieved a 31.7% improvement in performance:

**Final Performance**

![after](./images/after.png)

### CUDA Graph Impact

**Default Stream Performance**

![graph](./images/graph.png)

With all CUDA API calls (memset, memcpy, kernel...) on the default stream (implicitly synchronized), we cannot leverage on the parallel power of GPU hardware and between CPU and GPU. As shown in the above figure, the GPU can process only one call at a time, and the CPU has to wait for synchronization to execute the next CUDA API call. Besides, because our pipeline has several small tasks (~ 30us), the overhead of kernel launch is dominant. Both of these factors cause a lower GPU busy rate during each frame. 

<details>   <summary>Optimization: CUDA Graph</summary>  
	First, instead of copying back from GPU to decide the next kernel launch size, we launch a fixed size block for each kernel, which not only eliminates costly device-to-host mem copies but also enables the reuse of cuda graph each frame.
<br />
	Second, for inevitable host-to-device mem copies every frame (constants and buffer initialization), we use pinned memory and memcpyAsync to make it fit into our cuda graph execution.
<br />
	We built a dependency graph for all tasks and use different streams to process independent tasks with event recording for stream synchronization.
<br />
	<img src="./images/DAG.jpg" style="zoom:10%;" />
<br /> Dependency graph structure illustration
</details>



**Rewritten With Multi Stream & CudaGraph**

![graph](./images/graph.png)

**Performance Analysis**:

| Metric | Default Stream | CUDA Graph | Improvement |
|--------|--------------|------------|-------------|
| CPU Time | 7.79ms         | 4.2ms | +85.48% |
| GPU Time | 6.14ms         | 4.07ms | +50.85%     |
| GPU Time for Rasterize Pipeline | 6.02ms         | 2.81ms | +114.23%    |
| Total Frame Time | 7.79ms | 4.2ms      | +85.48% |

With cuda graph, we see a dramatic improvement in frame rate (from 120 to 240 FPS). The graph implementation optimizes the rendering pipeline by:

* Capturing the entire frame workload as a reusable graph
* Eliminating kernel launch overhead
* Enabling concurrent execution of independent operations

## Limitation & Failure Case

**Holes on model**

In release mode, there are some single-pixel size holes on the model. Since it's in release mode, we cannot use CUDA debugging to see what real happened there. Just some speculation: it may be caused by the precision of depth value - we store z-plane equations in float number and convert float depth to unsigned depth by multiplying a large float number, which can cause data precision problems when executed on different SMs at different time. 

![failure](./images/failure.png)

The cursor points out some holes on the bunny model when rendering in pure color.

**cuda graph limit**

OpenGL interop (`cudaGraphicsMapResources`) cannot be captured in graphs. So we have to write map and unmap logic outside the graph every frame. As these task takes a lot of time cpu time whiling keep the gpu idle, there is still a stall overhead waiting for the driver to complete resource mapping for gpu.

**kernel limit**

* CUDA kernels cannot directly use the L2 cache from the last kernel. So all buffers have to write back to global mem and then be fetched by other kernels, which requires more bandwidth than real hardware pipelines.
* Cooperative groups have a limitation on launched kernel block size, which makes our stream compaction unable to reach the theoretical maximum.

## Future Enhancements

**Render target compression**: Use 4:1 & 8:1 DCC for color and depth buffers, further reducing memory bandwidth for early/late-z and fragment write back 

**Compressed Textures**: Similar to RT compression, reduce bandwidth

**Cuda Graph Caching**: Like pipeline object caching, support rasterization with different settings and reduce overhead for capturing new graphs

## References

- [Real-Time Rendering, Fourth Edition, Chapter 23](https://www.realtimerendering.com/)
- [A trip through the Graphics Pipeline](https://fgiesen.wordpress.com/2011/07/09/a-trip-through-the-graphics-pipeline-2011-index/)
- [tiled caching rendering](https://www.realworldtech.com/tile-based-rasterization-nvidia-gpus/)
- [High-Performance Software Rasterization on GPUs](https://research.nvidia.com/sites/default/files/pubs/2011-08_High-Performance-Software-Rasterization/laine2011hpg_paper.pdf)
- S. Molnar, M. Cox, D. Ellsworth and H. Fuchs, "A sorting classification of parallel rendering," in *IEEE Computer Graphics and Applications*, vol. 14, no. 4, pp. 23-32, July 1994, doi: 10.1109/38.291528. keywords: {Sorting;Geometry;Concurrent computing;Computational efficiency;Hardware;Feedforward systems;Pipelines;Aggregates;Costs;Application software},
