# Continual-Reg
Welcome! This library provides the official implementation of our paper "Toward Universal Medical Image Registration via Sharpness-Aware Meta-Continual Learning" accepted by MICCAI 2024 [[arxiv](https://arxiv.org/abs/2406.17575)].

## Usage

1. Setup your dataset and its path in `./core/datasets/continual3d.py`.
2. Install the training tool [deep_kit](https://github.com/xzluo97/deep_kit/tree/dev). 
3. Setup the model configuration by modifying the yaml files in `./cfgs`.
4. Run the code in terminal by `tr mersam`. This would start the training using sharpness-aware meta-continual learning.
5. Test the trained model by running `te test_mersam`.

## Contact

For any questions or problems please [open an issue](https://github.com/xzluo97/Continual-Reg/issues/new) on GitHub.

## Citation

```bibtex
@inproceedings{ContinualReg,
  title={Toward Universal Medical Image Registration via Sharpness-Aware Meta-Continual Learning},
  author={Wang, Bomin and Luo, Xinzhe and Zhuang, Xiahai},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2024},
  organization={Springer}
}
```

