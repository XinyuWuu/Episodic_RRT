# Episodic RRT
This repository contains the official C++ implementation and demo for the Episodic RRT planner.

website: https://xinyuwuu.github.io/Episodic_RRT/

paper: https://arxiv.org/abs/2507.06605

code: https://github.com/XinyuWuu/Episodic_RRT (temporarily not available)

docker: https://hub.docker.com/r/xinyu0000/episodic_rrt (temporarily not available)

## Getting Started
You can run the demo using either Docker (recommended for a quick start) or by building the project from the source.

### Option 1: Run with Docker (Recommended)

1. Pull the Docker Image

```bash
docker pull xinyu0000/episodic_rrt
```

2. Run the Container

```bash
docker run --rm -it --network host xinyu0000/episodic_rrt:latest "/bin/bash"
```

> Note: The --network host flag is crucial. It allows the planner running inside the container to communicate with the [Rerun](https://rerun.io/) visualization client running on your local machine.

### Option 2: Build from Source

If you prefer to build the project manually, you must first set up the appropriate environment. Our development and testing were performed on Ubuntu 22.04.


1. Clone the Repository:
>```bash
>git clone https://github.com/XinyuWuu/Episodic_RRT.git
>cd Episodic_RRT
>```

2. Install Dependencies:
For all required dependencies and environment setup, please refer to the provided `Dockerfile`. It contains the complete list of packages (including [OpenVINO](https://docs.openvino.ai/2025/get-started/install-openvino.html)) and system configurations needed to build the project.


3. Build the Project:
The general build process uses CMake to configure and build the project from the cppenv directory into the build directory. For the exact commands, flags, and build targets, please see the `Dockerfile`.


### Running the Demo

After setting up the environment, follow these steps to run the planner and see the visualization.

1. Launch the Rerun Viewer:
First, open a new terminal and start the Rerun viewer. This will listen for data from the planner. If you don't have it installed, run:

> ⚠️ Note on Rerun Version: To ensure compatibility, please check the rerun-cpp version specified in `cppenv/CMakeLists.txt`. Your Python rerun-sdk version must match. You can install a specific version using pip install rerun-sdk==\<version\>.

>```bash
>pip install rerun-sdk
>rerun
>```

2. Run the Planner:
In your original terminal (inside the Docker container or your project directory), run the compiled test executable from the project's root directory:

> ```bash
> ./test
> ```

3. View the Visualization:
The planner will now attempt to connect to the Rerun viewer at `rerun+http://127.0.0.1:9876/proxy`. You should see the final path appear live in the Rerun window!


## Configuration
You can modify algorithm parameters or the Rerun viewer's IP address by editing `cppenv/test.cpp`. If you make changes, remember to re-compile the project.
