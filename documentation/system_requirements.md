# System requirements

## Operating System
DRAGON has been tested on Linux (Ubuntu 22.04), Amazon Web Services (ml.g4dn.2xlarge, ml.g5.xlarge) and MacOS. Windows is not supported, but may work (e.g., by using WSL2). I don't have access to a Windows machine, so I cannot give guidance on Windows-specific issues.

## Hardware requirements
We support GPU (recommended), CPU and Apple M1/M2 as devices.

### Hardware requirements for training
We recommend you use a GPU for training as this allows to use half precision training. Depending on the model and your hardware, training on CPU or MPS (Apple M1/M2) is reasonably fast.
When training with a GPU, one with at least 10 GB (popular non-datacenter options are the RTX 2080ti, RTX 3080/3090 or RTX 4080/4090) is required. Depending on the model you use (e.g., a `large` variant), more GPU memory is needed.
We also recommend a strong CPU to go along with the GPU, for example 6 cores (12 threads). Plus, the faster the GPU, the better the CPU should be!

### Hardware Requirements for inference
Again we recommend a GPU to make predictions as this will be faster and allows half precision. However, inference times are typically still manageable on CPU and MPS (Apple M1/M2).

### Example hardware configurations
Example workstation configurations for training (instances are from Amazon Web Servies, see online for more info!):

| Instance Size   | vCPUs | Instance Memory (GiB) | GPU Model   | GPUs | Total GPU memory (GB) | Memory per GPU (GB) | Instance Storage (GB) |
|-----------------|-------|-----------------------|-------------|------|-----------------------|---------------------|------------------|
| ml.g4dn.2xlarge | 8     | 32                    | NVIDIA T4   | 1    | 16                    | 16                  | 1 x 125 NVMe SSD      |
| ml.g5.xlarge    | 4     | 16                    | NVIDIA A10G | 1    | 24                    | 24                  | 1 x 250               |
