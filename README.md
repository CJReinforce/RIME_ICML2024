<!-- <h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> RIME: Robust Preference-based Reinforcement Learning with Noisy Preferences </h1> -->

# RIME: Robust Preference-based Reinforcement Learning with Noisy Preferences

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

<!-- ## Abstract

Preference-based Reinforcement Learning (PbRL) avoids the need for reward engineering by harnessing human preferences as the reward signal. However, current PbRL methods over-reliance on high-quality feedback from domain experts, which results in a lack of robustness. In this paper, we present RIME, a robust PbRL algorithm for effective reward learning from noisy preferences. Our method utilizes a sample selection-based discriminator to dynamically filter denoised preferences for robust training. To mitigate the accumulated error caused by incorrect selection, we propose to warm start the reward model, which additionally bridges the performance gap during the transition from pre-training to online training in PbRL. Our experiments on robotic manipulation and locomotion tasks demonstrate that RIME significantly enhances the robustness of the current state-of-the-art PbRL method. -->

## Requirements

### Installation

```bash
conda env create -f conda_env.yaml
conda activate rime
pip install -e .[docs,tests,extra]
cd custom_dmc2gym
pip install -e .
pip install git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld
pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

Install `mujoco` and `mujoco-py` according to [this link]( https://docs.google.com/document/d/1eBvfKoczKmImUgoGMbqypODBXmI1bD91/edit).

## Get Started

### Configs

Set hyperparameters in `run_parallel.sh`.

### Running

```bash
bash run_parallel.sh
```

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