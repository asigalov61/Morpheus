# Morpheus

## Most advanced fully open-sourced Symbolic MIDI Music AI implementation on GitHub

***

![the-art-warriors-morpheo](https://user-images.githubusercontent.com/56325539/147360073-59cfb940-9ed2-4903-8618-d3db58df3e24.jpg)

***

### Original Version
[![Open In Colab][colab-badge2]][colab-notebook2]

[colab-notebook2]: <https://colab.research.google.com/github/asigalov61/Morpheus/blob/main/%5BGC%5D_Morpheus.ipynb>
[colab-badge2]: <https://colab.research.google.com/assets/colab-badge.svg>

### 128x128 Version [WIP]
[![Open In Colab][colab-badge3]][colab-notebook3]

[colab-notebook3]: <https://colab.research.google.com/github/asigalov61/Morpheus/blob/main/Morpheus_128x128.ipynb>
[colab-badge3]: <https://colab.research.google.com/assets/colab-badge.svg>

***

## Main Features:

### 1) Most advanced Music AI technology to-date (GPT3+RPR[RGA]) with FULL(!) attention
### 2) Multiple-embedding technology (proper MIDI encoding with binary velocity)
### 3) 5-in-1 capabilities: performance, continuation, melody, accompaniment, inpainting(!)
### 4) Multi-channel MIDI capabilities (9 simulataneous MIDI instruments + drums)
### 5) Distributed training capabilities (easily train on multiple GPUs out of the box)
### 6) Pure PyTorch implementation (you only need PyTorch for training and inference)
### 7) Super-optimized and streamlined code (easy to understand and to modify)
### 8) BONUS: CLaMP capabilities (CLIP for Music)

***

### FAQ

### Q) How long should I train for?
### A1) Train for no more than 1 epoch. This usually works well. Training longer usually degrades performance.
### A2) You can try to cheat with the help of RPR and train only to full convergence (make sure to use random shuffling). But it is really dataset/task dependent so such trick may not always work for your particular purpose.

### Q) What is the idea behind Morpheus 128x128?
### A) We basically want to try to squeze music into symmetrical AND reasonable space. In this case its [127, 127, 127, 127*10, 1]. Music generally loves symmetry. So do the transformer NNs. Its not the most perfect arrangement, nor it is the most universal, but it does show better results over assymetrical encoding schemas.

### Q) Why Morpheus 128x128 does not use chordification?
### A) Chordification greately helps to save on train data size indeed. Unfortunatelly, this comes at a price: quality loss, especially on delicate datasets. Therefore, to allow for maximum music output quality Morpheus 128x128 excludes chordification.

***

### Project Los Angeles

### Tegridy Code 2022
