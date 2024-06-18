# MICCAI 2024 ResTL
The implementation of "Subject-Adaptive Transfer Learning Using Resting State EEG Signals for Cross-Subject EEG Motor Imagery Classification" early accepted at MICCAI 2024. (Top < 11%)

![Overview](https://github.com/SionAn/MICCAI2024-ResTL/assets/71419863/0a7fa98c-d056-40d4-bd65-6c352c526779)

## Run
* Example of training script
```
python main.py -g [gpu_id] -d [dataset] -m [model] -te [test subject] --checkpoint [path for saving model]
```
* To test the trained model, use -t 'test'

## Citation
```bibtex
@article{an2024subject,
  title={Subject-Adaptive Transfer Learning Using Resting State EEG Signals for Cross-Subject EEG Motor Imagery Classification},
  author={An, Sion and Kang, Myeongkyun and Kim, Soopil and Chikontwe, Philip and Shen, Li and Park, Sang Hyun},
  journal={arXiv preprint arXiv:2405.19346},
  year={2024}
}
```

## Thanks to
Jeon et al. : https://github.com/eunjin93/SICR_BCI \
EEG-Conformer: https://github.com/eeyhsong/EEG-Conformer