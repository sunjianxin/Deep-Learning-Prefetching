# RmdnCache: Prefetching Neural Network for Large-scale Volume Visualization

RmdnCache is a predictive deep neural network for microblock prefetching under multi-resolution large-scale volume visualization.
![results](https://github.com/sunjianxin/Deep-Learning-Prefetching/blob/main/teaser.jpg)
![results](https://github.com/sunjianxin/Deep-Learning-Prefetching/blob/main/teaser.png)
The detailed results can be visited from the demo video on Youtube from [here](https://youtu.be/SBPq6zV1LUQ)

# Experiment Setting

- Pytorch
- Jupyter notebook
- Nvidia GPU with CUDA supported
- Implemented neural network blocks: LSTM and MDN
- Transfer learning

# Data set
- Flame dataset: 7GB in size

# Training

Use the "training" folder for training the RmdnCache network. RNN should be trained first and then followed by MDN neural network training. 

# Inference

The input is the parameter of the POV of interest and the output is the predicted microblock indices for prefetching.
