# DuDoTrans
- This code is an official implementation of "[DuDoTrans: Dual-Domain Transformer for Sparse-View CT Reconstruction](https://link.springer.com/chapter/10.1007/978-3-031-17247-2_9)" based on the open source ct reconstruction toolbox [odl](https://github.com/odlgroup/odl).

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
Run the script on NIH-AAPM dataset.

`python main.py`

## Testing 
If  you want to test the model which has been trained on the NIH-AAPM dataset, turn off and "trainer.train()" and turn on "trainer.inference()", then download the pretrained model [here](https://drive.google.com/file/d/165KZKtZWxOTVb2ahHFpjW__Mli_tA13G/view?usp=share_link), put it under the subdirectory ./results/models/, and run the script as follows.

`python main.py`

## Citation
If you use our code or models in your work or find it is helpful, please cite the corresponding paper:

```
@inproceedings{wang2022dudotrans,
  title={DuDoTrans: Dual-Domain Transformer for Sparse-View CT Reconstruction},  
  author={Wang, Ce and Shang, Kun and Zhang, Haimiao and Li, Qian and Zhou, S Kevin},
  booktitle={International Workshop on Machine Learning for Medical Image Reconstruction},
  year={2022}
}
```
