# `dm_control`: DeepMind Infrastructure for Physics-Based Simulation.

DeepMind's software stack for physics-based simulation and Reinforcement
Learning environments, using MuJoCo physics.

An **introductory tutorial** for this package is available as a Colaboratory
notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/dm_control/blob/master/tutorial.ipynb)

## Overview

This package consists of the following "core" components:

-   [`dm_control.mujoco`]: Libraries that provide Python bindings to the MuJoCo
    physics engine.

-   [`dm_control.suite`]: A set of Python Reinforcement Learning environments
    powered by the MuJoCo physics engine.

-   [`dm_control.viewer`]: An interactive environment viewer.

Additionally, the following components are available for the creation of more
complex control tasks:

-   [`dm_control.mjcf`]: A library for composing and modifying MuJoCo MJCF
    models in Python.

-   `dm_control.composer`: A library for defining rich RL environments from
    reusable, self-contained components.

-   [`dm_control.locomotion`]: Additional libraries for custom tasks.

-   [`dm_control.locomotion.soccer`]: Multi-agent soccer tasks.

If you use this package, please cite our accompanying [tech report]:

```
@misc{tassa2020dmcontrol,
    title={dm_control: Software and Tasks for Continuous Control},
    author={Yuval Tassa and Saran Tunyasuvunakool and Alistair Muldal and
            Yotam Doron and Siqi Liu and Steven Bohez and Josh Merel and
            Tom Erez and Timothy Lillicrap and Nicolas Heess},
    year={2020},
    eprint={2006.12983},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}
```

## Requirements and Installation

`dm_control` is regularly tested on Ubuntu 16.04 against the following Python
versions:

*   3.6
*   3.7
*   3.8

Various people have been successful in getting `dm_control` to work on other
Linux distros, OS X, and Windows. We do not provide active support for these,
but will endeavour to answer questions on a best-effort basis.

Follow these steps to install `dm_control`:

1.  Download MuJoCo Pro 2.00 from the download page on the [MuJoCo website].
    MuJoCo Pro must be installed before `dm_control`, since `dm_control`'s
    install script generates Python [`ctypes`] bindings based on MuJoCo's header
    files. By default, `dm_control` assumes that the MuJoCo Zip archive is
    extracted as `~/.mujoco/mujoco200_$PLATFORM` where `$PLATFORM` is either
    `linux`, `win64`, or `macos`.

2.  Install the `dm_control` Python package by running `pip install dm_control`.
    We recommend `pip install`ing into a `virtualenv`, or with the `--user` flag
    to avoid interfering with system packages. At installation time,
    `dm_control` looks for the MuJoCo headers from Step 1 in
    `~/.mujoco/mujoco200_$PLATFORM/include`, however this path can be configured
    with the `headers-dir` command line argument.

3.  Install a license key for MuJoCo, required by `dm_control` at runtime. See
    the [MuJoCo license key page] for further details. By default, `dm_control`
    looks for the MuJoCo license key file at `~/.mujoco/mjkey.txt`.

4.  If the license key (e.g. `mjkey.txt`) or the shared library provided by
    MuJoCo Pro (e.g. `libmujoco200.so` or `libmujoco200.dylib`) are installed at
    non-default paths, specify their locations using the `MJKEY_PATH` and
    `MJLIB_PATH` environment variables respectively. These environment variables
    should be set to the full path to the relevant file itself, e.g.
    `export MJLIB_PATH=/path/to/libmujoco200.so`.

## Versioning

`dm_control` is released on a rolling basis: the latest commit on the `master`
branch of our GitHub repository represents our latest release. Our Python
package is versioned `0.0.N`, where `N` is the number that appears in the
`PiperOrigin-RevId` field of the commit message. We always ensure that `N`
strictly increases between a parent commit and its children. We do not upload
all versions to PyPI, and occasionally the latest version on PyPI may lag behind
the latest commit on GitHub. Should this happen, you can still install the
newest version available by running `pip install
git+git://github.com/deepmind/dm_control.git`.

## Rendering

The MuJoCo Python bindings support three different OpenGL rendering backends:
EGL (headless, hardware-accelerated), GLFW (windowed, hardware-accelerated), and
OSMesa (purely software-based). At least one of these three backends must be
available in order render through `dm_control`.

*   Hardware rendering with a windowing system is supported via GLFW and GLEW.
    On Linux these can be installed using your distribution's package manager.
    For example, on Debian and Ubuntu, this can be done by running `sudo apt-get
    install libglfw3 libglew2.0`. Please note that:

    -   [`dm_control.viewer`] can only be used with GLFW.
    -   GLFW will not work on headless machines.

*   "Headless" hardware rendering (i.e. without a windowing system such as X11)
    requires [EXT_platform_device] support in the EGL driver. Recent Nvidia
    drivers support this. You will also need GLEW. On Debian and Ubuntu, this
    can be installed via `sudo apt-get install libglew2.0`.

*   Software rendering requires GLX and OSMesa. On Debian and Ubuntu these can
    be installed using `sudo apt-get install libgl1-mesa-glx libosmesa6`.

By default, `dm_control` will attempt to use GLFW first, then EGL, then OSMesa.
You can also specify a particular backend to use by setting the `MUJOCO_GL=`
environment variable to `"glfw"`, `"egl"`, or `"osmesa"`, respectively. When
rendering with EGL, you can also specify which GPU to use for rendering by
setting the environment variable `EGL_DEVICE_ID=` to the target GPU ID.

## Additional instructions for Homebrew users on macOS

1.  The above instructions using `pip` should work, provided that you use a
    Python interpreter that is installed by Homebrew (rather than the
    system-default one).

2.  Before running, the `DYLD_LIBRARY_PATH` environment variable needs to be
    updated with the path to the GLFW library. This can be done by running
    `export DYLD_LIBRARY_PATH=$(brew --prefix)/lib:$DYLD_LIBRARY_PATH`.

[EXT_platform_device]: https://www.khronos.org/registry/EGL/extensions/EXT/EGL_EXT_platform_device.txt
[MuJoCo license key page]: https://www.roboti.us/license.html
[MuJoCo website]: http://www.mujoco.org/
[tech report]: https://arxiv.org/abs/2006.12983
[`ctypes`]: https://docs.python.org/3/library/ctypes.html
[`dm_control.mjcf`]: dm_control/mjcf/README.md
[`dm_control.mujoco`]: dm_control/mujoco/README.md
[`dm_control.suite`]: dm_control/suite/README.md
[`dm_control.viewer`]: dm_control/viewer/README.md
[`dm_control.locomotion`]: dm_control/locomotion/README.md
[`dm_control.locomotion.soccer`]: dm_control/locomotion/soccer/README.md
