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
```python Point_Cloud_Denoiser/Pointfilter/Customer_Module/chamfer_distance/setup.py install```


# Run Tests

```python -W ignore:ImportWarning -m unittest discover -s tests```  



to remove  

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install scipy plyfile scikit-learn matplotlib pillow=4.0.0 -c conda-forge
conda install -c conda-forge tqdm pyyaml easydict tensorboard pandas
conda install -c conda-forge point_cloud_utils==0.18.0
conda install -c fvcore -c iopath -c conda-forge fvcore iopath

pip freeze
