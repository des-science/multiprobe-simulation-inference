```
rsync -ahv --prune-empty-dirs \
    --include={"*/","*.yaml","*.h5","*.npy","*.pt","*.log","checkpoint/***"} \
    --exclude={"debug","**/wandb/**","*"} \
    /pscratch/sd/a/athomsen/deep_lss/ \
    /global/cfs/cdirs/des/athomsen/deep_lss/results
```

```
rsync -ahv --prune-empty-dirs \
  --exclude={"**/wandb/**","**/smoothing/**"} \
  --include={"*/","*.yaml","*.h5","*.npy","*.pt","*.tf","*.png","*.pdf"} \
  --exclude="*" \
  athomsen@perlmutter-p1.nersc.gov:/pscratch/sd/a/athomsen/deep_lss \
  /Users/arne/data/DESY3
```