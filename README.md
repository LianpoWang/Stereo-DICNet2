## Stereo-DICNet2
A unified and physics-guided speckle matching network for three-dimensional deformation measurement

## Dependencies
Stereo-DICNet2 is implemented in PyTorch and tested with Ubuntu, please install the dependencies using pip install.
* Python 3.9; 
* PyTorch;
* Torchvision;
* numpy;
* scipy;
* CUDA;
## File Structure
The Stereo-DICNet2/Network:
```
  ├── UniDIC                       # contains model for Stereo-DICNet2
  ├── datasetS                     # scripts to generate speckle image dataset
  ├── loss                         # scripts to Loss function design
  ├── utils                        # tool scripts
  ├── main.py                      # train the Stereo-DICNet2 model
  ├── test.py                      # test scripts
  └── README.md
```
## About dataset
Baidu cloud Link: https://pan.baidu.com/s/1bgAjIH4MTuX_hTk0wspTGg?pwd=xi9h <br/> 
Extract the code:xi9h
## Generate a dataset
First generate a dataset using python3 `Stereo-DICNet2/Network/datasets/SpeckleDataset.py`
## Train the model
Train the Stereo-DICNet2 model using python3 `Stereo-DICNet2/Network/main.py`
## Testing
Some testing scripts to check sanity of data and real tensile image samples `Stereo-DICNet2/Network/test.py`

## Results
Stereo-DICNet2 model measurement results:
![image](Stereo-DICNet2/Network/utils/result.png)

