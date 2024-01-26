source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda remove -n newtt --all -y
conda env create --solver=libmamba
conda activate newtt && cd /home/omar/Documents/mine/MY_LIBS/tictoc && pip install -e .
conda activate newtt && cd /home/omar/Documents/mine/MY_LIBS/Libraries && pip install -e .
conda activate newtt && cd /home/omar/Documents/mine/MY_LIBS/TreeTool && pip install -e .
conda activate newtt && cd /home/omar/Documents/mine/MY_LIBS/TreeTool
python test.py