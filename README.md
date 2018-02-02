# ClassicGAN

Generates classical MIDI files. Stacks 7 GANs, 3 levels deep.

## Getting Started

### Prerequisites

```
Tensorflow
tqdm
numpy
pretty-midi
matplotlib
```

### Installing

None necessary. Clone git, then run

```
python3 ClassicGAN.py
```

Put training datasets of your own under /Classics.

Modify, then run

```
python3 Convert.py
```

for visualizations of the generated results.

Conversion of generated results to midi files are being implemented.

## Deployment

Currently manual.

## TODOs

Integrate [Gradient Checkpointing](https://github.com/openai/gradient-checkpointing)

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
* **Amazon** - *Provider of funding*
* **Everyone Else I Questioned** - *Thanks!*
