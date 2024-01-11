# setup python environmnent
### pytorch
```
module load python/3.11

# find the path with module show pytorch
conda create --prefix /global/common/software/des/athomsen/torch_env --clone /global/common/software/nersc/pm-2022q4/sw/pytorch/2.0.1

python -m ipykernel install --user --name torch_env

pip install --force-reinstall --no-cache-dir healpy
pip install --force-reinstall --no-cache-dir sbi
pip install --force-reinstall --no-cache-dir icecream

# msfm, deep_lss, msi
pip install -e .
```
