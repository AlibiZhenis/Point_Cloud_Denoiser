git clone https://github.com/AlibiZhenis/Point_Cloud_Denoiser.git
Install Visual Studio Build Tools 2019

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install scipy plyfile scikit-learn matplotlib pillow=4.0.0 -c conda-forge
conda install -c conda-forge tqdm scipy scikit-learn pyyaml easydict tensorboard pandas
conda install -c conda-forge point_cloud_utils==0.18.0
conda install -c fvcore -c iopath -c conda-forge fvcore iopath

pip freeze
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install [wheel filename]


python Point_Cloud_Denoiser/Pointfilter/Customer_Module/chamfer_distance/setup.py install


nvidia rtx a4000
Windows 11
cuda 11.3
cudnn 8.2
python 3.7