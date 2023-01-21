from __future__ import absolute_import, division, print_function
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]=str(opts.cuda_device)
    os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
    os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

    from evaler import Evaler
    Evaler = Evaler(opts)
    Evaler.eval()