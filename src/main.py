import os, sys, torch, inspect, json, copy, argparse, random, config, distributed
import numpy as np 
from models.utils import get_model
from data.utils import get_dataset
from optim.base import train_base
#from optim.sparse import train_sparse
from optim.APTS_W import APTS_W
from optim.TR import TR
from torch.nn import functional as F
from torch import nn
import torch.distributed as dist
import subprocess
import torch.multiprocessing as mp

def prepare_distributed_environment(rank, master_addr, master_port, world_size):
    device_id = 0
    if rank is None and master_addr is None and master_port is None and world_size is None: # we are on a cluster
        ## Execute code on a cluster
        os.environ["MASTER_PORT"] = "29501"
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NNODES"]
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = os.environ["SLURM_NODEID"]
        node_list = os.environ["SLURM_NODELIST"]
        master_node = subprocess.getoutput(
            f"scontrol show hostname {node_list} | head -n1"
        )
        os.environ["MASTER_ADDR"] = master_node
        dist.init_process_group(backend="nccl")
    else: # we are on a PC
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port # A free port on the master node
        # os.environ['WORLD_SIZE'] = str(world_size) # The total number of GPUs in the distributed job
        # os.environ['RANK'] = '0' # The unique identifier for this process (0-indexed)
        # os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo" # "nccl" or "gloo"
        dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    device_id = dist.get_rank()
    print(f"Device id: {device_id}")

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


def main(rank, args, master_addr=None, master_port=None, world_size=None): 
    torch.backends.cuda.matmul.allow_tf32 = True # allows us to make sure we're able to use tensorfloat32 during training
    torch.backends.cudnn.allow_tf32 = True

    # prepare_distributed_environment(rank, master_addr, master_port, world_size)
    # args = {
    #     'master_addr': 'localhost',
    #     'master_port': '12345',
    #     'backend': 'gloo',
    #     'worldsize': 2
    # }

    args.rank = rank
    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)

    args.device = torch.device(args.device)
    device_type = "cuda" if "cuda" in str(args.device) else "cpu"
    if device_type == "cuda":
        torch.cuda.set_device(args.device)

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
    # model = distributed_backend.transform_model(model)

    # group_specs = distributed_backend.get_raw_model(model).get_parameter_group_specs()
    # param_name_mapping = {p_name: p for p_name, p in model.named_parameters()}
    # optimized_params_cnt = 0
    # for g in group_specs:
    #     params = []
    #     for p_name in g["params"]:
    #         translated_p_names = distributed_backend.translate_model_parameter_name_for_node(p_name)
    #         params += [param_name_mapping[p_name] for p_name in translated_p_names]
    #     g["params"] = params
    #     optimized_params_cnt += sum([p.numel() for p in g["params"]])
    # print("number of optimized parameters: %.3fM" % (optimized_params_cnt/1e6,))
    # print(len(torch.cat([p.flatten() for p in model.parameters()])))
    # if args.opt == 'adamw':
    #     use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
    #     print(f"using fused AdamW: {use_fused}")
    #     extra_args = dict(fused=True) if use_fused else dict()
    #     opt = torch.optim.AdamW(group_specs, lr=args.lr, betas=(args.beta1, args.beta2),
    #                             weight_decay=args.weight_decay, **extra_args)
    # elif args.opt == 'sgd':
    #     opt = torch.optim.SGD(group_specs, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # else: #APTS
    if rank==0:
        opt = APTS_W(model.parameters(), model, model2, **get_apts_w_params(nr_models=2))
    else:
        opt = APTS_W(model.parameters(), model, model2, **get_apts_w_params(nr_models=2))

    

    
    # if args.scheduler != 'none':
    #     if args.scheduler in ['cos', 'linear']:
    #         scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=opt, max_lr=args.lr, total_steps=args.iterations, 
    #                                                         pct_start=args.warmup_percent, anneal_strategy=args.scheduler, 
    #                                                         cycle_momentum=False, div_factor=1e2, final_div_factor=.05)
    #     else:
    #         raise NotImplementedError(f"Unknown scheduler type: {args.scheduler}.")
    # else:
    scheduler = None

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

    print(f"\nTraining model={args.model} \n{vars(args)}\n")

    stats = train(model, opt, data, scheduler, args.iterations, args.acc_steps, args.batch_size, args.sequence_length, 
                  eval_freq=args.eval_freq, 
                  distributed_backend=distributed_backend,
                  ckpt_path=f"{ckpt_path}/ckpt.pt", extra_args=args)
    
    args.device = None
    args.dtype = None
    stats['args'] = vars(args)
    if distributed_backend.is_master_process():
        with open(f"{ckpt_path}/summary.json", "w") as fs:
            json.dump(stats, fs)
    distributed_backend.finalize()


if __name__ == "__main__":
    cmd_args = get_args()
    model_parallel = True
    if not model_parallel:
        main(cmd_args)
    else:
        world_size = dist.get_world_size() if dist.is_initialized() else (torch.cuda.device_count() if torch.cuda.is_available() else 0)
        if world_size == 0:
            print("No CUDA device(s) detected.")
            exit(0)

        master_addr = 'localhost'
        master_port = '12345'
        cmd_args.master_addr = master_addr
        cmd_args.master_port = master_port
        cmd_args.world_size = world_size
        mp.spawn(main, args=(cmd_args, master_addr, master_port, world_size), nprocs=world_size, join=True)
