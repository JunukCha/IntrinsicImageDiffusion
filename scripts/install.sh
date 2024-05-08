conda env create -n iid -f environment.yml
conda activate iid
pip install stable-diffusion-sdkit==2.1.5 --no-deps
conda install xformers -c xformers