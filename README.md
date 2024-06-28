<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> RIME: Robust Preference-based Reinforcement Learning <br> with Noisy Preferences </h1>


<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://scholar.google.com/citations?user=IOiro9MAAAAJ" target="_blank" style="text-decoration: none;">Jie Cheng<sup>1,2</sup></a>&nbsp;,&nbsp;
    <a href="https://scholar.google.com/citations?user=F4ypDHIAAAAJ" target="_blank" style="text-decoration: none;">Gang Xiong<sup>1,2</sup></a>&nbsp;,&nbsp;
    <a href="https://scholar.google.com/citations?user=R4Rn7dMAAAAJ" target="_blank" style="text-decoration: none;">Xingyuan Dai<sup>1,2</sup></a>&nbsp;,&nbsp;
    <a href="https://scholar.google.com/citations?user=Q4B36ucAAAAJ" target="_blank" style="text-decoration: none;">Qinghai Miao<sup>2</sup></a>&nbsp;,&nbsp; 
    <a href="https://scholar.google.com/citations?user=RRKqjKAAAAAJ" target="_blank" style="text-decoration: none;">Yisheng Lv<sup>1,2</sup></a>&nbsp;&nbsp;
    <a href="https://scholar.google.com/citations?user=3TTXGAoAAAAJ" target="_blank" style="text-decoration: none;">Fei-Yue Wang<sup>1,2</sup></a>&nbsp;&nbsp;
	<br>
<sup>1</sup>State Key Laboratory of Multimodal Artificial Intelligence Systems, CASIA&nbsp;&nbsp;&nbsp;<br>
<sup>2</sup>School of Artificial Intelligence, the University of Chinese Academy of Sciences&nbsp;&nbsp;&nbsp;
</p>
<p align='center';>
<b>
<em>ICML 2024 Spotlight</em> <br>
</b>
</p>
<p align='center' style="text-align:center;font-size:2.5 em;">
<b>
    <a href="https://arxiv.org/abs/2402.17257" target="_blank" style="text-decoration: none;">[Paper]</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://github.com/CJReinforce/RIME_ICML2024" target="_blank" style="text-decoration: none;">[Code]</a>
</b>
</p>

## Requirements

### Install MuJoCo 2.1

```bash
sudo apt update
sudo apt install -y unzip gcc libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf libegl1 libopengl0
sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
mkdir ~/.mujoco
cd ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -zxvf mujoco210-linux-x86_64.tar.gz
rm -f mujoco210-linux-x86_64.tar.gz
```

Include the following lines in the `~/.bashrc` file:

```bash
export LD_LIBRARY_PATH=~/.mujoco/mujoco210/bin
export PATH="$LD_LIBRARY_PATH:$PATH"
```

Then run `source ~/.bashrc`

### Install dependencies

```bash
conda env create -f conda_env.yaml
conda activate rime
pip install -e .[docs,tests,extra]
cd custom_dmc2gym
pip install -e .
pip install git+https://github.com/rlworkgroup/metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb
pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

You could run `python -c "import mujoco_py; print(mujoco_py.__version__)"` to check if `mujoco-py` is installed properly. If not, see [FAQ](https://github.com/CJReinforce/RIME_ICML2024?tab=readme-ov-file#faq).

## Get Started

### Configs

Set hyperparameters in the all-in-one script `run_parallel.sh`, including the name of algorithm, hyperparameters of the algorithm and env, index of GPU for each random seed, etc.

### Running

For simulated (scripted) teachers:

```bash
bash run_parallel.sh
```

This will enable multi-threading to run experiments with multiple random seeds simultaneously.

For real human teachers (requires online annotation):
```bash
bash run_human_labeller.sh
```

When entering the annotation phase, run `label_program.ipynb` to annotate human preferences. The experimental result of RIME annotated by non-robotics students (detailed in Section 5.3) can be seen in [this GIF](https://raw.githubusercontent.com/CJReinforce/RIME_ICML2024/main/assets/hopper_backflip.gif).

## Acknowledgement

This repo benefits from [BPref](https://github.com/rll-research/BPref), [SURF](https://github.com/alinlab/SURF), [RUNE](https://github.com/rll-research/rune), and [MRN](https://github.com/RyanLiu112/MRN). Thanks for their wonderful work.

## Citation

```latex
@article{cheng2024rime,
  title={RIME: Robust Preference-based Reinforcement Learning with Noisy Preferences},
  author={Cheng, Jie and Xiong, Gang and Dai, Xingyuan and Miao, Qinghai and Lv, Yisheng and Wang, Fei-Yue},
  journal={arXiv preprint arXiv:2402.17257},
  year={2024}
}
```

## FAQ

1. `GLIBCXX_3.4.30` not found.

```bash
conda install gcc=12.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/env/rime/lib
```