# crystal-gnn

Major packages include tensorflow 2.2, pymatgen, and `nfp`.

nfp is the package @pstjohn put together, available at [nrel/nfp](https://github.com/nrel/nfp).

To re-create the inputs and trained model, first run `preprocess_crystals.py --config config/config.yaml`, then `train_model.py  --config config/config.yaml`.

I ran `preprocess_crystals.py` on the dav/login nodes (I'm bad), but submitted `train_model.py` via SLURM using `run.py --config config/config.yaml`.
