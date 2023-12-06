import os, sys, torch, inspect, json, copy, argparse, random, config, distributed
import numpy as np 
from models.utils import get_model
from data.utils import get_dataset
from optim.base import train_base
#from optim.sparse import train_sparse
from optim.APTS_W import APTS_W
from optim.TR import TR
from torch.nn import functional as F
import torch.distributed as dist
import subprocess
import torch.multiprocessing as mp


def get_apts_w_params(momentum=False, second_order=False, nr_models=2, max_iter=5, fdl=True, global_pass=True, device=None):
    TR_APTS_W_PARAMS_GLOBAL = {
        "radius": 0.01, #0.1,
        "max_radius": 0.1, #4.0
        "min_radius": 0.0001,
        "decrease_factor": 0.5,
        "increase_factor": 2.0,
        "is_adaptive": False,
        "second_order": second_order,
        "delayed_second_order": 0,
        "device": device,
        "accept_all": False,
        "acceptance_ratio": 0.75,
        "reduction_ratio": 0.25,
        "history_size": 5,
        "momentum": momentum,
        "beta1": 0.9,
        "beta2": 0.999,
        "norm_type": torch.inf,
    }

    TR_APTS_W_PARAMS_LOCAL = {
        "radius": 0.01,
        "max_radius": 0.1,
        "min_radius": 0,  # based on APTS class
        "decrease_factor": 0.5,
        "increase_factor": 2.0,
        "is_adaptive": False,
        "second_order": second_order,
        "delayed_second_order": 0,
        "device": device,
        "accept_all": False,
        "acceptance_ratio": 0.75,
        "reduction_ratio": 0.25,
        "history_size": 5,
        "momentum": momentum,
        "beta1": 0.9,
        "beta2": 0.999,
        "norm_type": torch.inf,
    }

    APTS_W_PARAMS = {
        "device": device,
        "max_iter": max_iter,
        "nr_models": nr_models,
        "global_opt": TR,
        "global_opt_params": TR_APTS_W_PARAMS_GLOBAL,
        "local_opt": TR,
        "local_opt_params": TR_APTS_W_PARAMS_LOCAL,
        "global_pass": global_pass,
        "forced_decreasing_loss": fdl,
        }

    return APTS_W_PARAMS

def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_format', default='base', choices=config.registered_formats())

    args, rem_args = parser.parse_known_args()

    return config.parse_args_with_format(format=args.config_format, base_parser=parser, args=rem_args, namespace=args)


def get_exp_name(args):
    """ Returns the name of the experiment, used for saving models. """
    exp_name = f"{args.model}_lr{args.lr}_bs{args.batch_size}x{args.acc_steps}_{args.world_size}nodes"
    if 'sparse' in args.model:
        exp_name += f"_lmd{args.lmbda}"
    exp_name += f"_seed={args.seed}"
    return exp_name


def main(rank=None, args=None, master_addr=None, master_port=None, world_size=None): 
    torch.backends.cuda.matmul.allow_tf32 = True # allows us to make sure we're able to use tensorfloat32 during training
    torch.backends.cudnn.allow_tf32 = True

    args.rank = int(os.environ["SLURM_NODEID"]) if rank == None else rank
    args.device = 'cuda:0' if args.distributed_backend == 'nccl' else torch.device(f'cuda:{args.rank}')
    args.device = torch.device(args.device)

    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)
    
    print(f"Rank {rank} args.device {args.device}")
    device_type = "cuda" if "cuda" in str(args.device) else "cpu"
    if args.distributed_backend == 'gloo' and device_type == "cuda":
       torch.cuda.set_device(args.device)
       torch.set_default_device(f'cuda:{rank}')

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Loading dataset '{args.dataset}'")
    
    data = get_dataset(args) # data is a dict: {'train': train_tokenized, 'val': eval_tokenized}
    if args.data_in_ram:
        data = {'train': np.array(data['train']), 'val': np.array(data['val'])}
        
    print(f"Num training tokens: {len(data['train'])}")
    print(f"Num validation tokens: {len(data['val'])}")
    
    model = get_model(args).to(args.device) # todo: take care of initializing the model if args.use_pretrained != 'none'
    torch.manual_seed(args.seed)
    model2 = get_model(args).to(args.device)
    
    opt = APTS_W(model.parameters(), model, model2, **get_apts_w_params(nr_models=args.world_size))

    args.world_size = distributed_backend.get_world_size()
    exp_name = get_exp_name(args)
    
    ckpt_path = os.path.join(args.results_base_folder, args.dataset, args.model, exp_name)
    if not os.path.exists(ckpt_path):
        if distributed_backend.is_master_process():
            os.makedirs(ckpt_path)
    elif os.path.isfile(os.path.join(ckpt_path, "summary.json")): # the experiment was already completed
        print(f"Already found experiment '{ckpt_path}'.\nSkipping.")
        sys.exit(0)

    if args.model == 'base': # all train functions have the same interface
        train = train_base
    #elif 'sparse' in args.model:
     #   train = train_sparse
    else:
        raise NotImplementedError(f"No training method implemented for model type '{args.model}'.")

    if args.rank == 0: 
        print(f"\nTraining model={args.model} \n{vars(args)}\n")

    scheduler = None
    stats = train(model, opt, data, scheduler, args.iterations, args.acc_steps, args.batch_size, args.sequence_length, 
                  eval_freq=args.eval_freq, 
                  distributed_backend=distributed_backend,
                  ckpt_path=f"{ckpt_path}/ckpt.pt", extra_args=args)
    
    args.device = None
    args.dtype = None
    stats['args'] = vars(args)
    if distributed_backend.is_master_process() and args.rank == 0:
        with open(f"{ckpt_path}/summary.json", "w") as fs:
            json.dump(stats, fs)
    distributed_backend.finalize()


if __name__ == "__main__":
    cmd_args = get_args() # Get default arguments
    try:
        node_list = os.environ["SLURM_NODELIST"]
    except:
        cmd_args.distributed_backend = 'gloo'
        
    if cmd_args.distributed_backend == 'nccl':
        node_list = os.environ["SLURM_NODELIST"]
        master_addr = subprocess.getoutput(
            f"scontrol show hostname {node_list} | head -n1"
        )
        cmd_args.world_size = int(os.environ["SLURM_NNODES"])
    elif cmd_args.distributed_backend == 'gloo':
        master_addr = 'localhost'
        cmd_args.world_size = torch.cuda.device_count()
    else:
        print("Unsupported distributed backend.")
        exit(0)
    
    if cmd_args.world_size == 0:
        print("No CUDA device(s) detected.")
        exit(0)

    os.environ['MASTER_PORT'] = '12345'  
    os.environ["MASTER_ADDR"] = master_addr

    cmd_args.master_addr = os.environ["MASTER_ADDR"]
    cmd_args.master_port = os.environ['MASTER_PORT']

    if cmd_args.distributed_backend == 'nccl':
        main(args=cmd_args, master_addr=cmd_args.master_addr, master_port=cmd_args.master_port, world_size=cmd_args.world_size)
    elif cmd_args.distributed_backend == 'gloo':
        mp.spawn(main, args=(cmd_args, cmd_args.master_addr, cmd_args.master_port, cmd_args.world_size), nprocs=cmd_args.world_size, join=True)
    else:
        print("Unsupported distributed backend.")
        exit(0)
    
