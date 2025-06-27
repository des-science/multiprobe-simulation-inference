# setup python environmnent
## pytorch
### new
See https://docs.nersc.gov/machinelearning/pytorch/#using-nersc-pytorch-modules
```
module load pytorch

pip install --user healpy sbi icecream trianglechain seaborn emcee esub-epipe numba sobol_seq tarp deprecation enflows flowcon

# msfm, deep_lss, msi
pip install -e .
```
### old
```
module load python/3.11

# find the path with module show pytorch
conda create --prefix /global/common/software/des/athomsen/torch_env --clone /global/common/software/nersc/pm-2022q4/sw/pytorch/2.0.1

python -m ipykernel install --user --name torch_env

pip install --force-reinstall --no-cache-dir healpy sbi icecream trianglechain seaborn emcee

# msfm, deep_lss, msi
pip install -e .
```
