# Run a baseline model in BasicTS framework.


import os
import sys
from argparse import ArgumentParser

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../.."))
sys.path.append(os.path.abspath(__file__ + '/../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import basicts
# TODO: remove it when basicts can be installed by pip



torch.set_num_threads(4) # aviod high cpu avg usage


def parse_args():
    parser = ArgumentParser(description="Run time series forecasting model in BasicTS framework!")
    parser.add_argument("-c", "--cfg", default="baselines/Normalization/Weather_tan.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/Normalization/ETTh1_sparsetsf.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/Normalization/Elec_nlinear.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/Normalization/ETTh1_cats.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/Normalization/ETTh1_tan.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/Normalization/PEMS08_tan.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/Normalization/ETTm1_tan.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/Normalization/ETTh1_Informer.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/Normalization/ETTh1_timemixer.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/Normalization/ETTh1_TimesNet.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/Normalization/ETTh1_nlinear.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/Normalization/ETTm1_tan.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/Normalization/ETTh1_nlinear.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/Normalization/ETTh1_sparsetsf.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/Normalization/ETTh1_patchtst.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/Normalization/ETTh1_umixer.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/Normalization/ETTh1_autoformer.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/iTransformer/ETTh1_tan.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/STID/METR-LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/STEP/STEP_METR-LA2.py", help="training config")
    #parser.add_argument("-c", "--cfg", default="baselines/Autoformer/Electricity.py", help="training config")
    #parser.add_argument("-c", "--cfg", default="baselines/Theta/Electricity.py", help="training config")
    #parser.add_argument("-c", "--cfg", default="baselines/Autoformer/Synth.py", help="training config")
    #parser.add_argument("-c", "--cfg", default="baselines/SparseTSF/Synth.py", help="training config")
    #parser.add_argument("-c", "--cfg", default="baselines/STID/Synth.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/DLinear/ExchangeRate.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/DLinear/Electricity.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/MDMixer/ETTh1.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/UMixer/ETTh1.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/NBeats/ETTh1.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/PatchTST/ETTh1.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/iTransformer/ETTh1_tan.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/iTransformer/ExchangeRate_tan.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/iTransformer/PEMS08_LTSF_tan.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/iTransformer/ExchangeRate.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/STF/Electricity.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/iTransformer/ETTh1.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/NLinear/ETTh1.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/SegRNN/ETTh1.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/STID/ETTh1.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/DLinear/ETTh1.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/NLinear/Electricity.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/NLinear/PEMS08_LTSF.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/NLinear/ETTm1.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/NLinear/ExchangeRate.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/STF/ExchangeRate.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/DLinear/PEMS08_LTSF.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/DLinear/ExchangeRate.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/DLinear/Synth.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/DGCRN/PEMS-BAY.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/DGCRN/example.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/complete_config.py", help="training config")
    parser.add_argument("-g", "--gpus", default="0", help="visible gpus")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(torch.cuda.is_available())
    basicts.launch_training(args.cfg, args.gpus, node_rank=0)
