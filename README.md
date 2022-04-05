# Morpheus

## Most advanced open-source Music AI implementation on GitHub


[![Open In Colab][colab-badge3]][colab-notebook3]

[colab-notebook3]: <https://colab.research.google.com/github/asigalov61/Morpheus/blob/main/%5BGC%5D_Morpheus.ipynb>
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

![the-art-warriors-morpheo](https://user-images.githubusercontent.com/56325539/147360073-59cfb940-9ed2-4903-8618-d3db58df3e24.jpg)

***

### FAQ

### Q) How long should I train for?
### A1) Train for no more than 1 epoch. This usually works well. Training longer usually degrades performance.
### A2) You can try to cheat with the help of RPR and train only to full convergence. It is really dataset/task depnendent so it may not always work.

### Q) What is the idea behind Morpheus Maker 128x128?
### A) We basically want to try to squeze music into symmetrical AND reasonable space. In this case its [127, 127, 127, 127*10, 1]. Music generally loves symmetry. So do the transformer NNs. Its not the most perfect arrangement, nor it is the most universal, but it does show better results over assymetrical encoding schemas.

***

### Project Los Angeles

### Tegridy Code 2022
