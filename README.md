# Stereo-DICNet2
A unified and physics-guided speckle matching network for three-dimensional deformation measurement

# Dependencies
Stereo-DICNet2 is implemented in PyTorch and tested with Ubuntu, please install the dependencies using pip install.
* Python 3.9; 
* PyTorch;
* Torchvision;
* numpy;
* scipy;
* CUDA;
# File Structure
```
 .. SDAS - 
```
# About dataset
Baidu cloud Link: https://pan.baidu.com/s/1bgAjIH4MTuX_hTk0wspTGg?pwd=xi9h <br/> 
Extract the code:xi9h
# Generate a dataset
First generate a dataset using python3 `Stereo-DICNet2/Network/datasets/SpeckleDataset.py`
# Train the model
Train the Stereo-DICNet2 model using python3 `Stereo-DICNet2/Network/main.py`
```
 .. SDAS -
 .. SDAS -
```  .
  ├── notebooks                             # contains ML model for DisplacementNet and notebook for testing ML model
  ├── dataset_generation                    # scripts to generate speckle image dataset
  ├── notebooks_archive                     # contains previously developed ML models to sanity check against
  ├── test                                  # test scripts and notebooks for sanity checking data
  ├── requirements.txt                      # install dependencies from this file
  └── README.md
# Testing
Some testing scripts to check sanity of data and real tensile image samples `Stereo-DICNet2/Network/test.py`

# Results
## Example 1
![image](result/res1.gif)
## Example 2
![image](result/res2.gif)

# Acknowledgement 
Part of our codes are adapted from PWC, UnFlow and UPflow, we thank the authors for their contributions.
