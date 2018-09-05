We are using TensorFlow 1.8.0 and Python 3.5.

It is strongly recommended installing these in the Anaconda Environment. Please refer to https://conda.io/docs/user-guide/install/index.html.

```shell
$ wget https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh
$ bash Anaconda3-5.1.0-Linux-x86_64.sh
```

Follow the prompts on the installer screens.
If you are unsure about any setting, accept the defaults. You can change them later.
To make the changes take effect, close and then re-open your Terminal window.


Once the anaconda is installed. You need to type the following commands to install TensorFlow in the virtual environment. Please refer to https://www.tensorflow.org/install/install_linux#InstallingAnaconda.

```shell
$ conda create -n tensorflow pip python=3.5
$ source activate tensorflow
$ pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.8.0-cp35-cp35m-linux_x86_64.whl
```