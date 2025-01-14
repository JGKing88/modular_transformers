# modular-transformers

Repository for the research I have conducted in Ev Fedorenko's lab at MIT. Initially this was a library that made it easy to build autoregressive transformers with by-layer architecture/hyper-parameter variability, but it has since become the repository where I explore the phenomena of trajectory straightening in LLMs (https://arxiv.org/abs/2311.04930).  Now the bulk of my research lives in scripts and is organized as follows:

## training_straightness
Training autoregressive transformers with weird architectures, variable hyper-parameters, and extra loss functions. Evaluating effects of these training features on internal curvature and output.  

## adding_straightness
Making internal model representations straighter by perturbing the activations with various methods. Evaluating effects of these perturbations on internal curvature and output.

## input_statistics
Training transformers with toy data generated with different statistical models to assess how the statistics of the training data affect the learned trajectory representations.

## dimensionality_reduction
Attempting to identify manifolds/subspaces/axes which vary in relation to trajectory curvature.

## attention_interpretability
Applying classic mechanistic interp techniques (ablations, attention visualizations, direct logit attribution) and a few curvature specific techniques to probe the mechanisms which are implicated in trajectory straightening.
