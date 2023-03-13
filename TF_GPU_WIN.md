## **Tensorflow-GPU Installation Guide on Windows:**

***⚠️ <ins>Warning</ins>: As of march 2023, tensorflow 2.11 isn't compatible with Windows GPU anymore so it won't detect your GPU. Instead, we recommend using tensorflow 2.10. Also, it seems that Visual Studio 2022 isn't compatible with the CUDA Toolkit so we recommend using the 2019 alternative. ⚠️***

If you want to install tensorflow on Windows, and use it with your GPU, you will need to follow these steps:

- Make sure your GPU is [CUDA-compatible](https://developer.nvidia.com/cuda-gpus). If your graphics card doesn't figure on this website, then you won't be able to train the network with your GPU.
- Go to [this website](https://www.tensorflow.org/install/source_windows#gpu) to check the compatibility of the versions of the software and modules we will be installing.
- First, [install](https://www.nvidia.com/Download/index.aspx) or update your GPU driver (to prevent issues, Nvidia highly recommends restarting your PC after any driver update or installation).
- Download and install [Visual Studio **2019**](https://visualstudio.microsoft.com/vs/older-downloads/). You will need it to compile the CUDA Toolkit. You shouldn't need to install any additional workloads or individual components.
- Download the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) with the correct version (depending on which vesion of tensorflow you're planning to use).
- Install CUDA with the Express installation. You don't need to launch anything at the end for the installation so just close the installer.
- Download [cuDNN](https://developer.nvidia.com/cudnn) with the correct version (depending on the version of tensorflow). NVIDIA forces you to have an account to download it, but it's free.
- Extract the files from the downloaded zip.
- Copy the three folders inside the zip (bin, include and lib).
- Paste them into <ins>C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X</ins> (replace the Xs with what version you're using. This should be where the CUDA files are stored). It will ask you if you want to replace the files in the destination to which you have to answer YES (for all the items).
- Copy the path of the bin folder and add it to your Environment Variables, inside the User variables' Path.
- Do the same for the "libnvvp" folder.
- Restart your computer to update the changes done on Environment Variables.
- Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (if you don't already have Anaconda installed)
- Launch Miniconda (the name of the command prompt is something like "Anaconda Prompt (Miniconda3)")
- Do the following commands (replace the python version with latest one that's compatible with the tensorflow version you're planning on using):

        conda create --name tf_X.X python==3.10
        conda activate tf_X.X

- If you want to install the latest version of tensorflow do the following command:

        pip install tensorflow

- If you want to install a specific version of tensorflow do the following command (replace the Xs with the version you want to use, e.g. tensorflow==2.10):

        pip install tensorflow==X.X

To make sure everything works, you can do the following commands in your created environment:

        python
        >>> import tensorflow as tf
        >>> len(tf.config.list_physical_devices('GPU'))

If it prints 1, your GPU is detected and you're good to go ! If it prints 0, then tensorflow doesn't detect your GPU, which means one of the previous steps was failed or there are new compatibility issues that we don't know about.