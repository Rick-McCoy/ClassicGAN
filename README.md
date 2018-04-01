# ClassicGAN

Generates classical MIDI files. Stacks 6 Generators, 4 levels deep, StackGAN++ style.

## Getting Started

### Prerequisites

```
tensorflow-gpu
tqdm
numpy
pretty-midi
matplotlib
toposort
networkx
pytest
```

Last three are for memory saving gradients.

### Installing

None necessary.

### Training

Git clone this repository, then run

```
python3 ClassicGAN.py
```

For your own datasets, put all midi files under /Classics or /TPD.

Then run

```
python3 Data.py
```

to convert them into .npy files.

**Warning**: These are extremely large, as on average, roughly 15 midi files gets converted into 1GB .npy files.

### Sampling

Make sure your checkpoints are all under /Checkpoints.

Select a midi file for encoding.

Then, run

```
python3 ClassicGAN.py -s /path/to/midi
```

Add ```-c True``` for concatenation of all generated 16 midi files.


## TODOs

Integrate [Gradient Checkpointing](https://github.com/openai/gradient-checkpointing) - Completed.

Convert generated results to midi - Completed.

Faster datasets using tf.data.Dataset - Completed.

Integrate VAEs - Completed.

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
