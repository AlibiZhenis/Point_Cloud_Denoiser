git clone https://github.com/AlibiZhenis/Point_Cloud_Denoiser.git
conda install -c conda-forge -c davidcaron pclpy
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install scipy plyfile scikit-learn matplotlib pillow=4.0.0 -c conda-forge
Install Visual Studio Build Tools 2019
pip install Cython chardet

Install PointCloudLibrary
https://python-pcl-fork.readthedocs.io/en/rc_patches4/install.html#install-python-pcl
https://www.studyplan.dev/pro-cpp/vcpkg-windows

conda install -c sirokujira pcl --channel conda-forge


python Point_Cloud_Denoiser/Pointfilter/Customer_Module/chamfer_distance/setup.py install


nvidia rtx a4000
cuda 11.3
cudnn 8.2
python 3.7