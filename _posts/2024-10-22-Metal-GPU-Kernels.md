# Unlocking the Power of Metal GPU on MacBooks: From Unified Memory to Practical Kernel Development in Rust

Apple's M Series chips have changed the game for high-performance computing, thanks to their **Metal GPU** and **unified memory architecture**. These technologies enable us to perform demanding tasks—like machine learning, data science, and real-time 3D rendering—entirely on local hardware, without needing external GPUs or cloud services.

In this blog, we’ll first dive into what **Metal GPU** and **unified memory** are and why they matter. Then, we’ll walk through a **practical example** of developing two Metal kernels in **Rust**—one to **square** a list of numbers and another to **cube** them—and expose them via a **Python wrapper** using the **PyO3** library. Finally, we'll explore how you can monitor your MacBook’s resource usage using **Asitop**, ensuring you’re getting the most out of your machine’s capabilities.

---

### Understanding Metal GPU and Unified Memory

#### Metal: Apple’s High-Performance Graphics and Compute API

**Metal** is Apple’s cutting-edge graphics and compute API, designed to provide us with low-level access to the capabilities of the GPU in Apple devices. While it was originally crafted for high-performance graphics rendering, it has evolved significantly to accommodate **General-Purpose GPU (GPGPU)** tasks, such as executing complex neural networks, simulations, and other compute-heavy applications.

##### Key Features of Metal GPU:
1. **Compute Shaders**: At the heart of Metal's functionality is the ability to write compute shaders, enabling us to run parallel computations on the GPU. This is particularly advantageous for tasks like machine learning, where we can leverage the GPU’s parallel processing capabilities to accelerate operations such as training and inference. For instance, when processing a batch of images, Metal allows us to perform thousands of computations concurrently, drastically reducing the time required to train a model.

2. **Unified API**: Metal serves as a cohesive framework across Apple’s platforms, including macOS, iOS, and tvOS. This unified approach simplifies our development workflow, allowing us to write code that works seamlessly across various Apple devices. Whether we’re developing an app for iPhones or macOS, Metal provides a consistent set of tools and features, which is crucial for maintaining a cohesive user experience.

3. **High Efficiency**: Metal is optimized specifically for Apple’s hardware. It grants us low-level access to GPU resources, empowering us to implement performance optimizations tailored to our applications. By minimizing overhead and maximizing throughput, we can achieve better performance in graphics-intensive applications, making our software more responsive and efficient.

#### Unified Memory Architecture: A Game Changer

One of the most transformative features of Apple’s silicon, particularly in the M-series chips, is the **unified memory architecture**. In traditional computing environments, CPUs and GPUs have distinct memory spaces, which necessitates data copying between them—a process that can create significant bottlenecks and slow down overall performance.

##### Benefits of Unified Memory:
- **Direct Access for CPU and GPU**: With unified memory, both the CPU and GPU can access the same memory pool. This eliminates the costly process of transferring data back and forth, allowing for immediate data manipulation and processing. For example, in machine learning scenarios, both the model and the data can reside in the same memory space, leading to faster access times and reduced latency.

- **Simplified Development**: This architecture streamlines our workflow by removing the need for complex memory management code. We no longer have to write intricate logic to handle memory transfers between the CPU and GPU, which simplifies our applications and reduces the risk of errors. 

- **Enhanced Performance**: The unified memory architecture significantly boosts performance for data-heavy tasks. By minimizing memory copying, we can reduce overhead and accelerate data processing. This is particularly beneficial in applications involving real-time data analysis, 3D rendering, and machine learning training.

### The Power of M-Series and A-Series Chips

Apple’s M-series chips (like the M1, M2, and their successors) and A-series chips (found in iPhones and iPads) are engineered with advanced GPU architectures that take full advantage of Metal.

#### M-Series GPUs:
The GPUs in M-series chips are designed for high parallelism and efficiency, featuring up to **10 GPU cores** in the M2 and beyond. These cores enable substantial computational power, allowing us to tackle demanding tasks such as video editing, game development, and machine learning directly on our devices. The M1 chip, for example, offers remarkable performance for GPU-intensive applications, making it an excellent choice for professionals who rely on computational power without the need for external GPUs.

- **Machine Learning Optimization**: M-series chips include specialized hardware for machine learning tasks, such as the **Neural Engine**, which accelerates tasks like image recognition and natural language processing. By leveraging Metal alongside the Neural Engine, we can achieve exceptional performance for AI-driven applications.

#### A-Series Bionic Chips:
Apple’s A-series chips, like the A15 Bionic, are also equipped with powerful GPUs that support advanced graphics and machine learning capabilities. These chips are designed for mobile devices, ensuring that even our smartphones and tablets can handle demanding tasks efficiently.

- **Real-Time Processing**: A-series chips feature technologies such as **Metal Performance Shaders**, which are optimized for running neural networks on iOS devices. This allows us to develop applications that perform real-time image processing, augmented reality experiences, and more—right from our mobile devices.

### Comparing Metal to NVIDIA GPUs

To better understand the strengths of Metal and Apple’s unified architecture, it’s essential to compare them to NVIDIA GPUs, which are widely recognized for their dominance in gaming and deep learning applications.

#### CUDA vs. Metal:
NVIDIA’s **CUDA** platform provides a comprehensive framework for general-purpose computing on their GPUs. It is widely used in scientific computing, deep learning, and high-performance computing. However, CUDA is less integrated into the operating system compared to Metal. While it offers robust tools for parallel processing, it requires a more complex setup, particularly when working across different devices or operating systems.

- **When to Use CUDA**: CUDA is ideal for applications requiring extensive parallel processing and complex numerical simulations. For instance, if we're working on large-scale deep learning projects using frameworks like TensorFlow or PyTorch that have strong CUDA support, we might lean towards NVIDIA GPUs.

- **When to Use Metal**: Metal shines in scenarios where we’re developing applications for the Apple ecosystem, especially when optimizing for performance on macOS or iOS devices. If our goal is to build a mobile application that uses machine learning features directly on an iPhone or iPad, Metal’s seamless integration and unified memory model make it the superior choice.

#### Memory Management:
NVIDIA GPUs typically require manual memory management. We need to transfer data explicitly between the CPU and GPU, which can create significant bottlenecks in performance. In contrast, Metal’s unified memory architecture allows us to share memory between CPU and GPU without manual transfers, greatly enhancing performance in data-intensive applications.

Let’s dive deeper into how Apple’s unified memory architecture (UMA) in the M-series chips, which powers both macOS and iOS devices, fundamentally changes memory management for high-performance applications, including machine learning and GPU-accelerated tasks.

### Unified Memory Architecture (UMA) in Apple Silicon

Traditionally, in systems powered by NVIDIA GPUs or Intel processors, memory is split between the CPU and GPU. The CPU typically has its own memory (RAM), and the GPU has its own dedicated video memory (VRAM). In such architectures, transferring data between CPU and GPU requires memory copies, which can be a performance bottleneck, especially for large datasets. However, Apple’s unified memory architecture (UMA) changes this dynamic significantly.

#### What is Unified Memory?

In Apple Silicon (M1, M2, and beyond), the CPU and GPU share the same physical memory. This means that data does not need to be copied between separate pools of memory when moving between the CPU and GPU. Both can access the same memory directly, eliminating the need for expensive memory transfers.

Key benefits of this architecture include:

1. **Shared Data Space**: There is a single memory pool accessible to both the CPU and GPU. This allows for faster collaboration between them, as data can be accessed and modified without duplication.
   
2. **Reduced Latency**: Since there’s no need to transfer data between CPU and GPU memory, UMA minimizes latency when switching between tasks that involve both compute and rendering processes. For example, in machine learning, the CPU can preprocess data and the GPU can run training algorithms without needing to copy large datasets back and forth.

3. **Optimized for Multitasking**: In a unified memory system, applications that heavily use both CPU and GPU (like ML or graphics-heavy applications) are highly efficient. Both processors access the same memory with low overhead, enabling more sophisticated real-time operations like training and inference on local machines.

#### How Apple Silicon Achieves This

Apple Silicon (M1, M2) integrates the CPU, GPU, and other components such as the Neural Engine onto a single chip, which significantly improves performance efficiency. By placing these components on the same chip, Apple can create a tightly integrated system where all parts share the same memory pool.

- **Shared Physical Memory**: In NVIDIA and Intel systems, there is a distinct physical memory for the CPU (typically DDR RAM) and GPU (GDDR or HBM memory). Apple bypasses this split by giving the CPU, GPU, and Neural Engine access to the same unified RAM.
  
- **Bandwidth Optimization**: Apple Silicon chips are designed with extremely high memory bandwidth (up to 400 GB/s on M2 Max). This means the GPU, Neural Engine, and CPU can access large datasets quickly without saturating the memory bandwidth, which is critical for tasks like machine learning training, where huge datasets need to be processed in real time.

#### Example: Memory Management in Metal vs CUDA

To highlight the difference between Apple's unified memory and traditional architectures like NVIDIA’s CUDA (which powers many ML frameworks), let's look at a practical scenario involving machine learning.

##### CUDA (Traditional Architecture)

In systems with discrete GPUs (such as NVIDIA), the CPU and GPU have separate memory pools:

1. **Host Memory (CPU)**: This is where the main application runs and holds datasets.
   
2. **Device Memory (GPU)**: Separate from CPU memory, this is where the GPU stores the data it processes.

When a task needs to be run on the GPU, data must be transferred from host (CPU) memory to device (GPU) memory:

- **Memory Transfer**: You must explicitly allocate GPU memory and copy data to and from it. For example, in CUDA, this is done using `cudaMalloc()` and `cudaMemcpy()`.

This memory copy introduces latency and complicates the workflow, particularly when processing large datasets (e.g., in machine learning where tensor data can be very large).

##### Metal (Unified Architecture)

In Apple Silicon’s unified memory architecture:

- **No Memory Transfer**: Since both the CPU and GPU share the same physical memory, there’s no need for copying data. This reduces overhead significantly.
  
- **Direct Access**: A Metal buffer created in shared memory can be accessed and modified by both the CPU and GPU directly. For example, when creating a tensor for machine learning operations, we don’t need to manage separate memory pools.

This dramatically simplifies development and boosts performance for high-performance computing tasks. Here’s how you might create a shared buffer in Metal (as demonstrated earlier):

```swift
let device = MTLCreateSystemDefaultDevice()
let size = 1024 * 1024 // 1 Million elements

// Create a buffer in unified memory
let buffer = device.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared)

// CPU writes to the buffer
let pointer = buffer.contents().assumingMemoryBound(to: Float.self)
for i in 0..<size {
    pointer[i] = Float(i) // Fill the buffer with data
}

// GPU can now access the same buffer without needing a copy
```

In CUDA, this same process would require explicit memory allocation and copying, adding latency that Apple’s unified memory architecture eliminates.

```cuda
float* h_data = new float[size];  // Host allocation
float* d_data;                    // Device allocation
cudaMalloc(&d_data, size * sizeof(float));
cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);
```

### The M-Series GPUs and Their Power

The M-series chips, including the M1 and M2, are equipped with highly efficient GPUs designed to handle heavy workloads. These GPUs are integrated on the same die as the CPU and share memory via the UMA. Here’s how Apple’s M-series GPUs stack up:

1. **High Core Count**: Apple Silicon GPUs come with high core counts. For instance, the M2 Pro has up to 38 GPU cores, capable of running complex compute tasks (e.g., ML inference, image processing, and real-time rendering).

2. **High Efficiency**: Thanks to the unified memory, there is very little overhead for memory access. This leads to high-efficiency compute operations with minimal bottlenecks.

3. **Parallel Compute Capabilities**: Just like NVIDIA’s CUDA cores, Apple’s GPU cores are designed to run highly parallel tasks. Metal’s compute shaders can perform similar GPGPU tasks that CUDA does on NVIDIA GPUs, making Apple Silicon suitable for ML and rendering tasks.

### A-Series Bionic Chips for iOS Development

The A-series chips (such as the A15 Bionic) in iPhones and iPads also benefit from the unified memory architecture. They leverage the same principles of tightly integrated CPU, GPU, and Neural Engine components, making these mobile devices capable of performing tasks that traditionally required desktops.

- **Neural Engine**: On A-series chips, Apple’s Neural Engine can handle over 15 trillion operations per second (TOPS), enabling on-device ML tasks like real-time image processing, object recognition, and more, while also benefiting from unified memory access.

### Parallel with NVIDIA and When to Use What

When comparing Apple’s Metal and unified memory architecture with NVIDIA’s CUDA and discrete GPU memory, the decision comes down to the type of application and ecosystem:

- **NVIDIA + CUDA**: Best for developers working with large-scale models on systems with discrete GPUs. NVIDIA GPUs are still leaders in specialized ML hardware, especially for cloud-based systems or large-scale AI projects. CUDA is mature and supported by many popular ML frameworks.

- **Apple Metal + M-Series**: Ideal for developers or ML researchers who want to work on local, energy-efficient devices (MacBooks, iPads). Unified memory provides a big advantage for on-device training and inference without worrying about memory transfers, making it great for mobile or desktop ML tasks.


### Leveraging Metal with MLX for Machine Learning

For those of us eager to tap into Metal’s GPU acceleration, **MLX (Machine Learning Exploration)** provides a rich ecosystem of libraries and examples that enable us to harness Metal’s capabilities directly on our Macs or iOS devices.

#### Key Features of MLX:
- **Practical Projects**: The [MLX-examples](https://github.com/ml-explore/mlx-examples) repository offers a variety of projects that demonstrate how to implement machine learning tasks using Metal. From image classification to object detection, these examples serve as a fantastic resource for both beginners and experienced developers looking to explore Metal’s potential.

- **Accelerated Learning**: MLX facilitates the training of deep neural networks locally on our machines, allowing us to leverage the GPU’s power to speed up learning processes. This is particularly useful for prototyping models quickly without relying on external cloud resources.

- **Optimized Resource Management**: By taking advantage of the unified memory architecture, MLX ensures our models can efficiently utilize available memory. This is crucial for handling larger datasets and complex models without facing memory constraints.

### Monitoring GPU and RAM Usage with Asitop

As we begin to leverage Metal’s GPU capabilities, monitoring system resource usage becomes vital for maximizing performance. **Asitop** is a powerful tool for Apple Silicon devices that allows us to monitor GPU utilization, RAM usage, and thermal pressure in real time.

#### Benefits of Using Asitop:
- **Resource Visualization**: Asitop provides insights into how much of our GPU and unified memory is being utilized during resource-intensive tasks. This is invaluable for optimizing our applications and ensuring they run smoothly without consuming unnecessary resources.

- **Thermal Monitoring**: Asitop helps us keep track of our machine’s thermal output, which is critical when running heavy workloads. By understanding our system’s thermal behavior, we can avoid thermal throttling and maintain consistent performance.

- **Performance Optimization**: By closely monitoring resource usage, we can adjust our applications for maximum efficiency, allowing us to become true power users of our MacBooks and iOS devices.

---


### Practical Example: Building Metal Kernels in Rust with a Python Wrapper

[Code Github Repo](https://github.com/akashicMarga/metal_kernels_tuts) 

Now that we’ve covered the theory behind Metal GPU and unified memory, let’s dive into a practical example. Here, we’ll build two simple Metal kernels in **Rust**—one to **square** a list of numbers and another to **cube** them—and expose them via a **Python wrapper** using the **PyO3** library. This example highlights how you can leverage Metal’s compute capabilities for GPU-accelerated tasks in your applications.

#### Step 1: Setting Up the Rust Project

Start by creating a new Rust project:

```bash
cargo new rust_metal_kernel --lib
cd rust_metal_kernel
```

Modify your `Cargo.toml` to include the necessary dependencies:

```toml
[dependencies]
metal = "0.24.0"
objc = "0.2"
pyo3 = { version = "0.18", features = ["extension-module"] }

[lib]
crate-type = ["cdylib"]
```

We’ll use the `metal` and `objc` crates to interface with Metal and its Objective-C API, while `pyo3` will be used to expose our functions to Python.

#### Step 2: Writing the Metal Kernels

Next, create a `square_kernel.metal` file in the `src` folder. This file will contain two kernels: one to square and one to cube numbers.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void square_kernel(const device float *in [[buffer(0)]],
                          device float *out [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
    out[id] = in[id] * in[id];
}

kernel void cube_kernel(const device float *in [[buffer(0)]],
                        device float *out [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
    out[id] = in[id] * in[id] * in[id];
}
```

- **square_kernel**: Squares each element of the input array.
- **cube_kernel**: Cubes each element of the input array.

#### Step 3: Writing the Rust Code

Now, update the `src/lib.rs` to implement Rust functions that load and execute both kernels:

```rust
use metal::*;
use pyo3::prelude::*;
use std::mem;

fn create_device() -> Device {
    Device::system_default().expect("No Metal device found")
}

fn run_kernel(kernel: &Function, input_data: Vec<f32>, output_size: usize) -> Vec<f32> {
    let device = create_device();
    let command_queue = device.new_command_queue();
    let pipeline_state = device
        .new_compute_pipeline_state_with_function(kernel)
        .expect("Failed to create pipeline state");

    // Create input buffer (shared memory between CPU and GPU)
    let input_buffer = device.new_buffer_with_data(
        input_data.as_ptr() as *const std::ffi::c_void,
        (input_data.len() * mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create output buffer (shared memory between CPU and GPU)
    let output_buffer = device.new_buffer(
        (output_size * mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline_state);
    encoder.set_buffer(0, Some(&input_buffer), 0); // Set input buffer at index 0
    encoder.set_buffer(1, Some(&output_buffer), 0); // Set output buffer at index 1

    let thread_group_size = 256; // Optimal thread group size
    let thread_count = MTLSize::new(output_size as u64, 1, 1);
    let threads_per_group = MTLSize::new(thread_group_size as u64, 1, 1);

    encoder.dispatch_threads(thread_count, threads_per_group);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Map the output buffer's contents back to Rust memory
    let ptr = output_buffer.contents() as *mut f32;
    let mut output_data = vec![0.0; output_size];
    unsafe {
        ptr.copy_to_nonoverlapping(output_data.as_mut_ptr(), output_size);
    }

    output_data
}

fn run_metal_square_kernel(input_data: Vec<f32>) -> Vec<f32> {
    let source = include_str!("square_kernel.metal"); // Load Metal shader source
    let device = create_device();
    let library = device
        .new_library_with_source(source, &CompileOptions::new())
        .expect("Failed to create Metal library");
    let kernel_function = library
        .get_function("square_kernel", None)
        .expect("Failed to get square kernel function");

    run_kernel(&kernel_function, input_data.clone(), input_data.len())
}

fn run_metal_cube_kernel(input_data: Vec<f32>) -> Vec<f32> {
    let source = include_str!("square_kernel.metal"); // Reuse the same source with cube function
    let device = create_device();
    let library = device
        .new_library_with_source(source, &CompileOptions::new())
        .expect("Failed to create Metal library");
    let kernel_function = library
        .get_function("cube_kernel", None)
        .expect("Failed to get cube kernel function");

    run_kernel(&kernel_function, input_data.clone(), input_data.len())
}

// Exposing both functions to Python
#[pyfunction]
fn square_numbers(input: Vec<f32>) -> PyResult<Vec<f32>> {
    Ok(run_metal_square_kernel(input))
}

#[pyfunction]
fn cube_numbers(input: Vec<f32>) -> PyResult<Vec<f32>> {
    Ok(run_metal_cube_kernel(input))
}

// Python module setup
#[pymodule]
fn rust_metal_kernel(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(square_numbers, m)?)?;
    m.add_function(wrap_pyfunction!(cube_numbers, m)?)?;
    Ok(())
}
```

In this code:
- `create_device`: Initializes the Metal device.
- `run_kernel`: A helper function that sets up the Metal command buffer and executes a given kernel.
- `run_metal_square_kernel` and `run_metal_cube_kernel`: Load the respective kernels and call `run_kernel` to process the data.
- `square_numbers` and `cube_numbers`: Functions exposed to Python that can be called with a list of floats.

#### Step 4: Building the Python Extension

Use **maturin** to build the Rust project as a Python extension:

```bash
pip install maturin
maturin develop
```

This command compiles the Rust project into a Python module, making it available for use in Python.

#### Step 5: Using the Python Wrapper

Create a Python script (`test.py`) to call both functions:

```python
import rust_metal_kernel

input_data = [1.0, 2.0, 3.0, 4.0]

squared_output = rust_metal_kernel.square_numbers(input_data)
print(f"Squared Output: {squared_output}")

cubed_output = rust_metal_kernel.cube_numbers(input_data)
print(f"Cubed Output: {cubed_output}")
```

Running this script will output:

```plaintext
Squared Output: [1.0, 4.0, 9.0, 16.0]
Cubed Output: [1.0, 8.0, 27.0, 64.0]
```

---

### Conclusion

With **Metal GPU** and **unified memory**, Apple Silicon brings unparalleled efficiency and performance to tasks traditionally relegated to external GPUs or cloud services. Through tools like **MLX**, we can harness this power locally to build machine learning models, while tools like **Asitop** allow them to monitor resource usage and optimize performance.

Our example demonstrates how easy it is to write Metal kernels in **Rust** and expose them through a **Python** interface using **PyO3**. Whether you're performing heavy computations or building machine learning pipelines, Metal allows you to unlock the full potential of your MacBook—transforming it into a high-performance workstation for all your computing needs.

For further exploration, check out:
- [MLX Examples on GitHub](https://github.com/ml-explore/mlx-examples)
- [Asitop for performance monitoring](https://github.com/tlkh/asitop)
- [Metal Programming Guide by Apple](https://developer.apple.com/metal/)
- [Metal Puzzels](https://github.com/abeleinin/Metal-Puzzles) 
The Metal-Puzzles repository is an educational project that provides a collection of small programming exercises designed to teach GPU programming using Apple’s Metal API. The project is structured as a series of puzzles where users implement Metal shaders to solve specific computational problems, allowing them to explore the capabilities of Metal in a hands-on way.

--- 
