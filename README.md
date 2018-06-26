# ClassicGAN

A GAN for generating classical music. Works with piano rolls of midi files. Basic structure is a mixture of MuseGAN and StackGAN++.

## Getting Started

### Prerequisites

```
tensorflow-gpu
tqdm
numpy
pretty-midi
matplotlib
```

### Installing

None necessary. Clone this repository.

### Dataset Generation

For your own datasets, put all midi files under /Classics.

Then run

```
python3 Data.py
```

to convert them into .tfrecord files.

Efficient compression has been integrated, tfrecord files are now generated at a ratio of approximately 1:100.
(Ex: 220MB of midi files are converted into a single 20GB tfrecord file.)

Note: these tfrecord files are highly compressible, and the entire tfrecord file compresses to a 56MB .7z file.

### Training

Run

```
python3 ClassicGAN.py
```

For timeline file generations, add ```-r```.

For concatenation of all generated midi files, add ```-c```.

### Sampling

Make sure your checkpoint is under /Checkpoints.

Select a midi file for encoding.

Then, run

```
python3 ClassicGAN.py -s /path/to/midi
```

## TODOs

~~Integrate [Gradient Checkpointing](https://github.com/openai/gradient-checkpointing) - Completed.~~ Removed after benchmarks showed no significant speedup/memory advantage.

Convert generated results to midi - Completed.

Faster datasets using tf.data.Dataset - Completed.

~~Integrate VAEs - Completed.~~ Removed after training instability.

TTUR - Completed.

Currently tinkering with Spectral Normalization.

## Authors

* **Rick-McCoy** - *Initial work* - [Rick-McCoy](https://github.com/Rick-McCoy)

## License

This project is licensed under the MIT License.

## Acknowledgments

* **Jin-Jung Kim** - *General structure of code adapted* - [golbin](https://github.com/golbin)
* **Ishaan Gulrajani** - *Loss function code used* - [igul222](https://github.com/igul222)
* **MuseGAN: Multi-track Sequential Generative Adversarial Networks for Symbolic Music Generation and Accompaniment, Hao-Wen Dong et al.** - *Idea of stacking generators* - [link](http://arxiv.org/abs/1709.06298v2)
* **StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks, Han Zhang et al.** - *Idea of multiple levels* - [link](http://arxiv.org/abs/1710.10916v2)
* **Improved Training of Wasserstein GANs, Ishaan Gulrajani et al.** - *WGAN-GP used.* - [link](http://arxiv.org/abs/1704.00028v3)
* **SPECTRAL NORMALIZATION FOR GENERATIVE ADVERSARIAL NETWORKS, Takeru Miyato et al.** - *Spectral normalization* - [link](http://arxiv.org/abs/1802.05957v1)
* **GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium, Martin Heusel et al.** - *TTUR* - [link](http://arxiv.org/abs/1706.08500v6)
* **Nhat M. Nguyen** - *Spectral Normalization code used* - [minhnhat93](https://github.com/minhnhat93)
* **Amazon** - *Provider of funding*
* **Everyone Else I Questioned** - *Thanks!*
