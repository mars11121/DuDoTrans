# DuDoTrans
- This code is an official implementation of "DuDoTrans: Dual-Domain Transformer for Sparse-View CT Reconstruction" based on the open source ct reconstruction toolbox [odl](https://github.com/odlgroup/odl).

## Installation

## Requirements
- platform: linux-64
- python=3.6.13
- CUDA 11.0 or higher

## Data Acquisition
- The “NIH-AAPM-Mayo Clinic Low Dose CT Grand Challenge” dataset ([NIH-AAPM 2017](https://www.aapm.org/grandchallenge/lowdosect/)) could be acquired from [here](https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h). Note that the download link is updated 2021, and our experimental data is chosen before and differs from the current one, while this doesn't affect the model comparison.
- The COVID-19 dataset is in-house dataset.

## Data Preparation (NIH-AAPM)
After downloading the dataset from [here](https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h), put the train/test data (original dcm files) in the corresponding workdir "path to train" and "path to test".

## Training
Run the training script on NIH-AAPM dataset.

`python train.py`

## Testing 
If  you want to test the model which has been trained on the NIH-AAPM dataset, run the testing script as following.

`python test.py`

# Citation
If you use our code or models in your work or find it is helpful, please cite the corresponding paper:

- **DuDoTrans**:
```
@inproceedings{wang2022dudotrans,
  title={DuDoTrans: Dual-Domain Transformer for Sparse-View CT Reconstruction},  
  author={Wang, Ce and Shang, Kun and Zhang, Haimiao and Li, Qian and Zhou, S Kevin},
  booktitle={International Workshop on Machine Learning for Medical Image Reconstruction},
  year={2022}
}
```
