## XFeat-ONNX: ONNX Accelerated Features for Lightweight Image Matching
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![ONNX](https://img.shields.io/badge/ONNX-grey)](https://onnx.ai/)
### [[ArXiv]](https://arxiv.org/abs/2404.19174) | [[Project Page]](https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/) |  [[CVPR'24 Paper]](https://cvpr.thecvf.com/)
Open Neural Network Exchange (ONNX) compatible implementation of [XFeat: Accelerated Features for Lightweight Image Matching](https://github.com/verlab/accelerated_features/tree/main). The ONNX model format allows for interoperability across different platforms with support for multiple execution providers, and removes Python-specific dependencies such as PyTorch.
Source code in C++ is also provided to test the model!.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Inference](#inference)
- [Caveats](#caveats)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation
TODO or Pip install something.

## Usage
### Inference
TODO

## Caveats
As the ONNX Runtime has limited support for features like dynamic control flow, certain configurations of the models cannot be exported to ONNX easily. These caveats are outlined below.

### Feature Extraction
- Only batch size `1` is currently supported. This limitation stems from the fact that different images in the same batch can have varying numbers of keypoints, leading to non-uniform (a.k.a. *ragged*) tensors.
For this reason the code differ from the original project: batch operations have been removed.

### Multiscale
- Actually Dense Multiscale model is not supported. Possible Future Work!

### Keypoints location
- To compare the Pytorch/Onnx/C++ models, the images in the assets folder were used. It can be seen in the results that the Python Pytorch/ONNX results are very similar to each other. For the C++ part, the results may vary slightly, particularly with a noticeable upward sliding of points compared to the Python versions. However, when using the images without applying padding (image_800x608), the sliding does not occur. Could some scaling/rounding factor be getting lost somewhere?

## Citation
Project taken from: [XFeat](https://github.com/verlab/accelerated_features/tree/main)

Please cite the paper:
```bibtex
@INPROCEEDINGS{potje2024cvpr,
  author={Guilherme {Potje} and Felipe {Cadar} and Andre {Araujo} and Renato {Martins} and Erickson R. {Nascimento}},
  booktitle={2024 IEEE / CVF Computer Vision and Pattern Recognition (CVPR)}, 
  title={XFeat: Accelerated Features for Lightweight Image Matching}, 
  year={2024}}
```

## License
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

## Acknowledgements
Ty for the amazing job!
[Guilherme Potje](https://guipotje.github.io/) · [Felipe Cadar](https://eucadar.com/) · [Andre Araujo](https://andrefaraujo.github.io/) · [Renato Martins](https://renatojmsdh.github.io/) · [Erickson R. Nascimento](https://homepages.dcc.ufmg.br/~erickson/)
