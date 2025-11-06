from experiments.setups import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    dry_run = True

    ecl_96.run(dry_run=dry_run)
    ettm1_96.run(dry_run=dry_run)
    etth1_96.run(dry_run=dry_run)
