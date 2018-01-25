# ClassicGAN

Generates classical MIDI files. Stackes 17 generators to simulate various instruments.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

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
* **MuseGAN: Multi-track Sequential Generative Adversarial Networks for Symbolic Music Generation and Accompaniment, Hao-Wen Dong et al.** - *Idea of stacking generators* - [link](http://arxiv.org/abs/1709.06298v2)
* **Amazon** - *Provider of funding*
* **Everyone Else I Questioned** - *Thanks!*
