# System & Environment
GPU: NVIDIA RTX A4000  
OS: WIndows 11  
Python 3.7  
CUDA 11.3, cuDNN 8.2  

# Setup

1. Install Visual Studio Build Tools 2019  
2. Create conda environment  
```conda env create -f environment.yml```
3. Install more dependencies using pip  
```pip install -r requirements.txt```
4. Install pytorch3d and torch_cluster using wheel file  
```pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"```  
```pip install torch_cluster-1.6.0+pt112cu113-cp37-cp37m-win_amd64.whl```  
5. Setup Chamfer distance (for evaluation)  
```cd Pointfilter/Customer_Module/chamfer_distance```
```python setup.py install```
6. Download PointCleanNet pretrained model
```python pointcleannet/models/download_models.py --task denoising```
7. Setup DMR
```cd DMRDenoise/ops/emd```
```python setup.py install```


# Run Tests

```python -W ignore:ImportWarning -m unittest discover -s tests```  

# Usage example

Usage example can be found in eval.ipynb