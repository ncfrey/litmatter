# LitMatter
A template for rapid experimentation and scaling deep learning models on molecular and crystal graphs.

## How to use
1. Clone this repository and start editing, or save it and use it as a template for new projects.
2. Edit `lit_models/models.py` with the PyTorch code for your model of interest. 
3. Edit `lit_data/data.py` to load and process your PyTorch datasets.
4. Perform interactive experiments in `prototyping.py`.
5. Scale network training to any number of GPUs using the example batch scripts.

## Principles
LitMatter uses [PyTorch Lightning](https://zenodo.org/record/3828935#.YSP51x0pC-F) to organize PyTorch code so scientists can rapidly experiment with geometric deep learning and scale up to hundreds of GPUs without difficulty. Many amazing applied ML methods (even those with open-source code) are never used by the wider community because the important details are buried in hundreds of lines of boilerplate code. It may require a significant engineering effort to get the method working on a new dataset and in a different computing environment, and it can be hard to justify this effort before verifying that the method will provide some advantage. Packaging your code with the LitMatter template makes it easy for other researchers to experiment with your models and scale them beyond common benchmark datasets.

## Features
* Maximum flexibility. LitMatter supports arbitrary PyTorch models and dataloaders.
* Eliminate boilerplate. Engineering code is abstracted away, but still accessible if needed.
* Full end-to-end pipeline. Data processing, model construction, training, and inference can be launched from the command line, in a Jupyter notebook, or through a SLURM job.
* Lightweight. Using the template is *easier* than not using it; it reduces infrastructure overhead for simple and complex deep learning projects.

## Examples
The example notebooks show how to use LitMatter to scale model training for different applications.
* [Prototyping GNNs](./prototyping.ipynb) - train an equivariant graph neural network to predict quantum properties of small molecules.
* [Neural Force Fields](./LitNFFs.ipynb) - train a neural force field on molecular dynamics trajectories of small molecules.
* [DeepChem](./LitDeepChem.ipynb) - train a PyTorch model in DeepChem on a MoleculeNet dataset.
* [🤗](./LitHF.ipynb) - train a 🤗 language model to generate molecules.  

Note that these examples have additional dependencies beyond the core depdencies of LitMatter.

## References
If you use LitMatter for your own research and scaling experiments, please cite the following work:
[Frey, Nathan C., et al. "Scalable Geometric Deep Learning on Molecular Graphs." NeurIPS 2021 AI for Science Workshop. 2021.](https://arxiv.org/abs/2112.03364)
```
@inproceedings{frey2021scalable,
  title={Scalable Geometric Deep Learning on Molecular Graphs},
  author={Frey, Nathan C and Samsi, Siddharth and McDonald, Joseph and Li, Lin and Coley, Connor W and Gadepally, Vijay},
  booktitle={NeurIPS 2021 AI for Science Workshop},
  year={2021}
}
```

Please also cite the relevant frameworks: [PyG](https://arxiv.org/abs/1903.02428), [PyTorch Distributed](https://arxiv.org/abs/2006.15704), [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning),

and any extensions you use:
[🤗](https://arxiv.org/abs/1910.03771), [DeepChem](https://github.com/deepchem/deepchem#citing-deepchem), [NFFs](https://github.com/learningmatter-mit/NeuralForceField#references), etc.

## Extensions
When you're ready to upgrade to fully configurable, reproducible, and scalable workflows, use [hydra-zen](https://github.com/mit-ll-responsible-ai/hydra-zen). hydra-zen [integrates seamlessly](https://mit-ll-responsible-ai.github.io/hydra-zen/how_to/pytorch_lightning.html) with LitMatter to self-document ML experiments and orchestrate multiple training runs for extensive hyperparameter sweeps.

## Environment
Version management in Python is never fun and deep learning dependencies are always changing, but here are the latest tested versions of key dependencies for *LitMatter*
* Python 3.8
* Pytorch Lightning 1.5.1
* Pytorch 1.10.0

## Disclaimer

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited. 

© 2021 MASSACHUSETTS INSTITUTE OF TECHNOLOGY

    Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014)
    SPDX-License-Identifier: MIT

This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.

The software/firmware is provided to you on an As-Is basis.

