# OpenNMT-py for Summarization

## Requirements

All dependencies can be installed via:

```bash
pip install -r requirements.txt
```

Note that we currently only support PyTorch 1.0.0


### Step 1: Preprocess the data

```bash
sh preprocess.sh
```

We will be working with some example data in `data/` folder.

The data consists of parallel source (`src`) and target (`tgt`) data containing one sentence per line with tokens separated by a space:

* `src-train.txt`
* `tgt-train.txt`
* `src-val.txt`
* `tgt-val.txt`

Validation files are required and used to evaluate the convergence of the training. It usually contains no more than 5000 sentences.


After running the preprocessing, the following files are generated:

* `dataset.train.pt`: serialized PyTorch file containing training data
* `dataset.valid.pt`: serialized PyTorch file containing validation data
* `dataset.vocab.pt`: serialized PyTorch file containing vocabulary data


Internally the system never touches the words themselves, but uses these indices.

### Step 2: Train the model

```bash
sh train.sh
```

### Step 3: Test the model

```bash
sh test.sh
```
### Step 4: Check Tensorboard Loss

```bash
tensorboard --logdir=data/tensorboard/pointer/yelp/
```

## Acknowledgements

OpenNMT-py is run as a collaborative open-source project.

OpentNMT-py belongs to the OpenNMT project along with OpenNMT-Lua and OpenNMT-tf.

## Citation

[OpenNMT: Neural Machine Translation Toolkit](https://arxiv.org/pdf/1805.11462)

[OpenNMT technical report](https://doi.org/10.18653/v1/P17-4012)

```
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {Open{NMT}: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```
