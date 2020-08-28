# recommendation-tensorflow-2

This is implementation for some recommendation algorithms using tensorflow 2:

    - FISM: Factored Item Similarity Models for Top-N Recommender Systems, https://tsinghua-nslab.github.io/seminar/2013Autumn/8_11/FISM-paper.pdf
    - NAIS: Neural Attentive Item Similarity Model for Recommendation, https://arxiv.org/pdf/1809.07053.pdf

The implementation uses ideas from paper:

    - BPR: Bayesian Personalized Ranking from Implicit Feedback https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf
    
for pair-wise loss. And using confidence for implicit feedback from paper:

    - Collaborative Filtering for Implicit Feedback Datasets: http://yifanhu.net/PUB/cf.pdf


## Quick to Start

try FISM.ipynb for example

## Environment

Python 3.6

TensorFlow >= 2.0.0

Numpy >= 1.16

PS. For your reference, our server environment is Intel Xeon CPU E5-2630 @ 2.20 GHz and 64 GiB memory. We recommend your free memory is more than 16 GiB to reproduce our experiments (and we are still trying to reduce the memory cost...).

## Dataset

We provide two processed datasets: MovieLens 1 Million (ml-1m)

train.csv:

- Train file.
- Each Line is a training instance: user_id,item_id,rating

test.csv:

- Test file (positive instances).
- Each Line is a testing instance: user_id,item_id,rating

There are 10000 sample in test.csv which are randomly selected from ml-1m.

Update: August 28, 2020
