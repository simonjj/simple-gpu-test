# Using GPU on Azure Container Apps

## Introduction

Azure Container Apps provides a app-focused view to application developers. This means infrastructure is managed and not directly exposed to users. This works great for almost all workloads since containers abstract much of the details away. With GPU workload profiles this assumption no longer applies. In order for you to use the GPUs which come with your workload profile you will have to ensure certain libraries are in sync between the underlying operating system and the workload container. This article provide guidance on how you can build and test such a container.

> **Note:** At publish time we recommend the use of image `nvidia/cuda:12.1.1-runtime-ubuntu22.04` throughout this tutorial. This may change as Azure Container App's GPU workload fleet evolves and we recommend you [check alternative versions](https://hub.docker.com/r/nvidia/cuda/tags) in case you run into problems with the above.

## Setup

For specific details on how to use Azure Container App's workload profiles please see [Manage workload profiles with the Azure CLI](https://learn.microsoft.com/en-us/azure/container-apps/workload-profiles-manage-cli?tabs=external-env&pivots=aca-vnet-managed). For this exercise it is important to remember that provisioning a GPU workload profile requires the creation of a Azure Container Apps environment with the `----enable-dedicated-gpu --enable-workload-profiles` flags. This will automatically select a region with GPU availability and allocate a profile with the name `gpu` for you.

After your environment has been provisioned you can place applications/containers onto the GPU workload profile through the use of the `--workload-profile-name gpu`. Here's a quick example on how you can standup an environment and application we'll be using in our next step:

```
# define some basic variables
export ACA_ENV=cuda-test
export RG=cuda-playground
export LOCATION=westus3
export CUDA_IMAGE=nvidia/cuda:12.1.1-runtime-ubuntu22.04

# create a resource group
az group create -l $LOCATION --name $RG

# create an environment with gpu workload profile
az containerapp env create \
    --name $ACA_ENV \
    --resource-group $RG \
    --location $LOCATION \
    --enable-dedicated-gpu \
    --enable-workload-profiles
```

Now that we have the environment we run a Nvidia image which is prepped with CUDA. At the time of this writing `12.1.1` is a reasonable compromise between the most recent and oldest available CUDA toolkit versions. **The important thing here is that we match the CUDA driver (which lives on the host) with the CUDA Toolkit (which lives in the container)**. 

```
# standup a basic cuda enabled container
az containerapp create \
    --name cuda-explorer \
    --resource-group $RG \
    --environment $ACA_ENV \
    --workload-profile-name gpu \
    --cpu 4 --memory 8Gi \
    --image $CUDA_IMAGE \
    --command-line "/bin/bash, -c, sleep 12h" \
    --min-replicas 1 --max-replicas 1
```

## Finding the Underlying CUDA Details

After the above setup you should now have access to the `cuda-explorer` app running on your GPU workload profile. This application runs an official [Nvidia image](https://hub.docker.com/r/nvidia/cuda) with the needed toolkit, libraries and runtime included to leverage CUDA. For our purpose of investigating the underlying workload profile host we simply run the `sleep` as a container "workload". Next we run the `nividia-smi` tool to display the CUDA driver and toolkit versions installed.

```
# run the following command
az containerapp exec -g $RG --name cuda-explorer --command nvidia-smi
```

`nvidia-smi` is similar to `top` but for your GPU. You should see similar output: 

```
Wed Apr 17 22:01:19 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.54.03              Driver Version: 535.54.03    CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100 80GB PCIe          On  | 00000001:00:00.0 Off |                    0 |
| N/A   28C    P0              40W / 300W |      4MiB / 81920MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

If you see something similar to the above output your workload profile is ready to use. We can see the CUDA Toolkit version (top right) is `12.2`. The driver version (top middle) is `535.54.03`. At the bottom of the output you can see the the current processes running on the GPU (`No running processes found`). 

**Should you see a message about a failure to initialize NVML it is a due to `nvidia-smi` being unable to communicate with the driver. There's likely a driver-toolkit mismatch**. Try to (**upgrade to a newer tag**)[https://hub.docker.com/r/nvidia/cuda/tags] by following [these steps](#rebuilding-the-tester-image). For more detailed information on which driver is needed for which toolkit see [Minor Version Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#minor-version-compatibility).


## Exercising the GPU

As the last step we want to exercise the CPU to confirm our application is actually able to leverage the GPU, and compare how much faster than on CPU. You can do so by deploying a pre-build container from `simonj.azurecr.io/gpu-tester`. This container [includes a simple](https://github.com/simonjj/simple-gpu-test/blob/main/gpu_cpu_tester.py) [PyTorch](https://pytorch.org/) script which executes a sizeable matrix multiplication. First on the CPU and then on the GPU to show the delta between the two. 

```
# standup the gpu tester
az containerapp create \
    --name gpu-tester \
    --resource-group $RG \
    --environment $ACA_ENV \
    --workload-profile-name gpu \
    --cpu 2 --memory 4Gi \
    --image simonj.azurecr.io/gpu-tester \
    --min-replicas 1 --max-replicas 1
```

If you check the log output of the application, the above application should produce the following output to stdout:

```
2024-04-19T23:27:18.372559359Z CUDA is available. Running on GPU.
2024-04-19T23:27:18.372588373Z Result:
2024-04-19T23:27:18.372593042Z tensor([[  96.2215,  110.3856,   98.8350,  ...,   92.6685,  194.6051,
2024-04-19T23:27:18.372596829Z          -151.6151],
2024-04-19T23:27:18.372599474Z         [ 205.5347,  -66.7743,   65.3369,  ...,  101.4372,  -98.7581,
2024-04-19T23:27:18.372601928Z            60.8353],
2024-04-19T23:27:18.372604443Z         [ 137.5099,  -65.0038,  259.1740,  ...,  284.2198,  174.8975,
2024-04-19T23:27:18.372606918Z          -135.4074],
2024-04-19T23:27:18.372609402Z         ...,
2024-04-19T23:27:18.372611907Z         [ -53.5666,  -29.2805, -103.1653,  ...,  232.8224, -135.2330,
2024-04-19T23:27:18.372614352Z           271.0094],
2024-04-19T23:27:18.372616827Z         [-194.0985,  251.4910, -250.8734,  ...,  -79.4291, -117.2673,
2024-04-19T23:27:18.372619261Z           141.3860],
2024-04-19T23:27:18.372621766Z         [ 194.9410,   32.1693,  -61.0127,  ...,  -75.4219,  203.4215,
2024-04-19T23:27:18.372624180Z            13.1645]])
2024-04-19T23:27:18.372627306Z Elapsed Time: 3.227062255859375 seconds
```

## Rebuilding the Tester Image

The referenced image can also be rebuild from [`https://github.com/simonjj/simple-gpu-test`](https://github.com/simonjj/simple-gpu-test) using the steps below. Please note that we're also setting a `TARGET_IMAGE` here.

```
# we assume docker is installed
# checkout the code
git clone https://github.com/simonjj/simple-gpu-test
cd simple-gpu-test

# decide on the new location/name of the image
export TARGET_IMAGE=yourcr.azurecr.io/your-image-name
# build and push
docker build --build-arg BASE_IMAGE=nvidia/cuda:12.1.1-runtime-ubuntu22.04  -t $TARGET_IMAGE .
docker push $TARGET_IMAGE
```