# GrokAlign: Geometric Characterisation and Acceleration of Grokking

[Paper](https://arxiv.org/abs/2506.12284)

This is the official GitHub repository for our paper "GrokAlign: Geometric Characterisation and Acceleration of Grokking". `utils.py` provides classes to perform GrokAlign, compute centroids, and compute the explained variance of the first principal component of the Jacobian matrix on a model for input batches.

- `comparison_to_ntk.py`: Code for the experiment of Section 4.1, with the default configuration values reproducing the data for Figure 3.
- `xor_grokking.py`: Code for the experiment of Section 4.2, with the default configuration values reproducing the data for the top row of Figure 4, and the default values with `lambda_jac=1.0` reproducing the data for the bottom row of Figure 4.
- `delayed_robustness.py`: Code for the experiment of para "Inducing Delayed Robustness." of Section 4.3, with the default configuration values reproducing the data for the "Weight-Decay" run and the default configuration values with `lambda_jac=0.001` reproducing the data for the "GrokAlign" run of Figure 5.
- `inhibiting_generalisation.py`: Code for the experiment of para "Inhibiting Delayed Generalisation." of Section 4.3, with the defauly configuration values reproducing the data for the "Minimised Jacobian" run and the default configuration values with `jac_level=5.0` reproducing the data for "Constrained Jacobian" run of Figure 6.
- `accelerating_grokking.py`: Code for the experiment of para "Accelerating Grokking." of Section 4.3, with the default configuration values reproducing the "Baseline": setting `lambda_jac=1e-3`,`loss_fn='CrossEntropy` reproduces the "GrokAlign" run under the cross-entropy loss function, setting `lambda_jac=1e-4`,`loss_fn='MSE` reproduces the "GrokAlign" run under the mean-squared error loss function, utilising `--grokfast` reproduces the Grokfast run, and utilising `--adv_training` reproduces the "Adversarial Training" run.
- `transformer_alignment.py`: Code for the experiment of para "Controlling the Learned Solutions of Deep Networks." of Section 4.3, with the defauly configuration values reproducing the "Weight-Decay" run of Figure 7, the default configuration values with `lambda_jac=1e-3` reproducing the "GrokAlign" run of Figure 7, and utilising `--fixed_embedding` reproduces the corresponding runs for Figure 8.

## Citation

    @article{,
      title={GrokAlign: Geometric Characterisation and Acceleration of Grokking},
      author={Walker, Thomas and Humayun, Ahmed Imtiaz and Balestriero, Randall and Baraniuk, Richard},
      journal={arXiv},
      year={2025}
    }
