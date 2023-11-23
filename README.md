<h2> 
<a href="https://whu-usi3dv.github.io/Mobile-Seed/" target="_blank">Mobile-Seed: Joint Semantic Segmentation and Boundary Detection for Mobile Robots</a>
</h2>

This is the official PyTorch implementation of the following publication:

> **Mobile-Seed: Joint Semantic Segmentation and Boundary Detection for Mobile Robots**<br/>
> [Youqi Liao](https://martin-liao.github.io/), [Shuhao Kang](https://scholar.google.com/citations?user=qB6B7lkAAAAJ&hl=zh-CN&oi=sra), [Jianping Li](https://jianping.xyz/), [Yang Liu](https://mruil.github.io/), [Yun Liu](https://yun-liu.github.io/), [Zhen Dong](https://dongzhenwhu.github.io/index.html), [Bisheng Yang](https://3s.whu.edu.cn/info/1025/1415.htm),[Xieyuanli Chen](https://xieyuanli-chen.com/),<br/>
> *ArXiv 2023*<br/>
> [**Paper**](https://arxiv.org/abs/2311.12651) | [**Project-page**](https://whu-usi3dv.github.io/Mobile-Seed/) | **Video**


## 🔭 Introduction
<p align="center">
<strong>TL;DR: Mobile-Seed is an online framework for simultaneous semantic segmentation
and boundary detection on compact robots.</strong>
</p>
<img src="utils/media/motivation.png" alt="Motivation" style="zoom:50%;">

<p align="justify">
<strong>Abstract:</strong> Precise and rapid delineation of sharp boundaries
and robust semantics is essential for numerous downstream
robotic tasks, such as robot grasping and manipulation, realtime semantic mapping, and online sensor calibration performed on edge computing units. Although boundary detection
and semantic segmentation are complementary tasks, most
studies focus on lightweight models for semantic segmentation but overlook the critical role of boundary detection. In
this work, we introduce Mobile-Seed, a lightweight, dual-task
framework tailored for simultaneous semantic segmentation
and boundary detection. Our framework features a two-stream
encoder, an active fusion decoder (AFD) and a dual-task regularization approach. The encoder is divided into two pathways:
one captures category-aware semantic information, while the
other discerns boundaries from multi-scale features. The AFD
module dynamically adapts the fusion of semantic and boundary information by learning channel-wise relationships, allowing for precise weight assignment of each channel. Furthermore,
we introduce a regularization loss to mitigate the conflicts in
dual-task learning and deep diversity supervision. Compared to
existing methods, the proposed Mobile-Seed offers a lightweight
framework to simultaneously improve semantic segmentation
performance and accurately locate object boundaries. Experiments on the Cityscapes dataset have shown that Mobile-Seed
achieves notable improvement over the state-of-the-art (SOTA)
baseline by 2.2 percentage points (pp) in mIoU and 4.2 pp
in mF-score, while maintaining an online inference speed of
23.9 frames-per-second (FPS) with 1024×2048 resolution input
on an RTX 2080 Ti GPU. Additional experiments on CamVid
and PASCAL Context datasets confirm our method’s generalizability.
</p>

## 🆕 News
- 2023-11-22: [[Project page]](https://whu-usi3dv.github.io/Mobile-Seed/) (with introduction video) is aviliable! 🎉
- 2023-11-22:  [[Preprint paper]](https://arxiv.org/abs/2311.12651) is aviliable! 🎉

## 💡 Citation
If you find this repo helpful, please give us a star~.Please consider citing Mobile-Seed if this program benefits your project
```
@article{liao2023mobileseed,
  title={Mobile-Seed: Joint Semantic Segmentation and Boundary Detection for Mobile Robots},
  author={Youqi Liao and Shuhao Kang and Jianping Li and Yang Liu and Yun Liu and Zhen Dong and Bisheng Yang and Xieyuanli Chen},
  journal={arXiv preprint arXiv:2311.12651},
  year={2023}
}
```

## 🔗 Related Projects
We sincerely thank the excellent projects:
- [AFFormer](https://github.com/dongbo811/AFFormer) for head-free Transformer;
- [SeaFormer](https://github.com/fudan-zvg/SeaFormer) for Squeeze-enhanced axial Transformer;
