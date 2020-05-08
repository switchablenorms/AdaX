from __future__ import print_function, absolute_import
import torch
import numpy as np
from math import pow, pi, cos
import shutil, os, glob
import torch.distributed as dist
import multiprocessing as mp

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best=False, save_path='checkpoint', step=0):
    filepath = os.path.join(save_path, 'ckpt_step{}.pth'.format(step))
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_best.pth'))


class LRScheduler(object):
    r"""Learning Rate Scheduler
    For mode='step', we multiply lr with `decay_factor` at each epoch in `step`.
    For mode='poly'::
        lr = targetlr + (baselr - targetlr) * (1 - iter / maxiter) ^ power
    For mode='cosine'::
        lr = targetlr + (baselr - targetlr) * (1 + cos(pi * iter / maxiter)) / 2
    If warmup_epochs > 0, a warmup stage will be inserted before the main lr scheduler.
    For warmup_mode='linear'::
        lr = warmup_lr + (baselr - warmup_lr) * iter / max_warmup_iter
    For warmup_mode='constant'::
        lr = warmup_lr
    Parameters
    ----------
    mode : str
        Modes for learning rate scheduler.
        Currently it supports 'step', 'poly' and 'cosine'.
    niters : int
        Number of iterations in each epoch.
    base_lr : float
        Base learning rate, i.e. the starting learning rate.
    epochs : int
        Number of training epochs.
    step : list
        A list of epochs to decay the learning rate.
    decay_factor : float
        Learning rate decay factor.
    targetlr : float
        Target learning rate for poly and cosine, as the ending learning rate.
    power : float
        Power of poly function.
    warmup_epochs : int
        Number of epochs for the warmup stage.
    warmup_lr : float
        The base learning rate for the warmup stage.
    warmup_mode : str
        Modes for the warmup stage.
        Currently it supports 'linear' and 'constant'.
    """

    def __init__(self, optimizer, max_steps, lr_mult=1.0, args=None):
        super(LRScheduler, self).__init__()

        self.mode = args.lr_type
        self.warmup_mode = args.warmup_mode if hasattr(args, 'warmup_mode') else 'linear'
        assert (self.mode in ['step', 'poly', 'cosine', 'linear', 'multi_cosine'])
        assert (self.warmup_mode in ['linear', 'constant'])

        self.optimizer = optimizer

        self.base_lr = args.base_lr if hasattr(args, 'base_lr') else 0.1
        self.base_lr *= lr_mult
        self.learning_rate = self.base_lr

        self.step = [int(i) for i in args.step.split(',')] if hasattr(args, 'step') else [100, 150]
        self.iteration = [int(i) for i in args.step.split(',')] if hasattr(args, 'iterations') else [150000, 300000, 450000]
        self.decay_factor = args.decay_factor if hasattr(args, 'decay_factor') else 0.1
        self.targetlr = args.targetlr if hasattr(args, 'targetlr') else 0.00001
        self.targetlr *= lr_mult
        self.power = args.power if hasattr(args, 'power') else 2.0
        self.warmup_lr = args.warmup_lr if hasattr(args, 'warmup_lr') else 0.0
        self.warmup_lr *= lr_mult
        self.max_iter = max_steps
        self.warmup_iters = args.warmup_steps if hasattr(args, 'warmup_steps') else 0.0
        self.warmup_iters /= lr_mult

    def update(self, step):
        T = step
        assert T >= 0 # and T <= self.max_iter)

        if self.warmup_iters > T:
            # Warm-up Stage
            if self.warmup_mode == 'constant':
                self.learning_rate = self.warmup_lr
            else:  # if self.warmup_mode == 'linear':
                self.learning_rate = self.warmup_lr + (self.base_lr - self.warmup_lr) * \
                                     T / self.warmup_iters
        else:
            self.learning_rate = self.targetlr
            if self.mode == 'step':
                count = sum([1 for s in self.step if s <= step])
                if type(self.decay_factor) == list:
                    for i in range(count):
                        self.learning_rate *= self.decay_factor[i]
                else:
                    self.learning_rate = self.base_lr * pow(self.decay_factor, count)
            elif self.mode == 'poly':
                self.learning_rate = self.targetlr + (self.base_lr - self.targetlr) * \
                                     pow(1 - (T - self.warmup_iters) / (self.max_iter - self.warmup_iters), self.power)
            elif self.mode == 'cosine':
                self.learning_rate = self.targetlr + (self.base_lr - self.targetlr) * \
                                     (1 + cos(pi * (T - self.warmup_iters) / (self.max_iter - self.warmup_iters))) / 2
            elif self.mode == 'linear':
                self.learning_rate = self.base_lr - (self.base_lr - self.targetlr) * \
                                     ((T - self.warmup_iters) / (self.max_iter - self.warmup_iters))
            elif self.mode == 'multi_cosine':
                for i in range(len(self.iteration)):
                    if T < self.iteration[i]:
                        self.learning_rate = self.targetlr + (self.base_lr - self.targetlr) * \
                                     (1 + cos(pi * (T - self.iteration[i-1] - self.warmup_iters) / (self.iteration[1] - self.warmup_iters))) / 2
                        break
            else:
                raise NotImplementedError

        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.learning_rate
        return self.learning_rate



class ColorAugmentation(object):
    def __init__(self, eig_vec=None, eig_val=None):
        if eig_vec == None:
            eig_vec = torch.Tensor([
                [0.4009, 0.7192, -0.5675],
                [-0.8140, -0.0045, -0.5808],
                [0.4203, -0.6948, -0.5836],
            ])
        if eig_val == None:
            eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])
        self.eig_val = eig_val  # 1*3
        self.eig_vec = eig_vec  # 3*3

    def __call__(self, tensor):
        assert tensor.size(0) == 3
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val):
        if self.length > 0:
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val
            self.count += 1
            self.avg = self.sum / self.count


def get_last_weights_old(path):
    files = glob.glob(path)
    if len(files) == 0:
        return ''
    files.sort(key=lambda fn: os.path.getmtime(fn) if not os.path.isdir(fn) else 0)
    print("Trained ckpts:")
    if len(files) > 5:
        print('[...]', files[-5:])
    else:
        print(files)
    return files[-1]


def get_last_weights(path):
    files = glob.glob(path)
    if len(files) == 0:
        return ''
    files_step = [int(f.split('/')[-1].strip('ckpt_step').strip('.pth')) for f in files]
    files_step = sorted(files_step)
    # print(files_step)
    del files
    files = [path.replace('*', str(s)) for s in files_step]
    print("Trained ckpts:")
    if len(files) > 5:
        print('[...]', files[-5:])
    else:
        print(files)
    return files[-1]


def get_weights_list(path, sort=True, start_step=0):
    files = glob.glob(path)
    if len(files) == 0:
        return ''
    files_step = [int(f.split('/')[-1].strip('ckpt_step').strip('.pth')) for f in files]
    files_step_temp = []
    for step in files_step:
        if step >= start_step:
            files_step_temp.append(step)
        else:
            pass
    files_step = files_step_temp.copy()
    del files_step_temp
    if sort:
        files_step = sorted(files_step)
    else:
        pass
    del files
    files = [path.replace('*', str(s)) for s in files_step]
    print("Eval ckpt:", files)
    return files


def dist_init(port):
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id%num_gpus)

    if '[' in node_list:
        beg = node_list.find('[')
        pos1 = node_list.find('-', beg)
        if pos1 < 0:
            pos1 = 1000
        pos2 = node_list.find(',', beg)
        if pos2 < 0:
            pos2 = 1000
        node_list = node_list[:min(pos1,pos2)].replace('[', '')
    addr = node_list[8:].replace('-', '.')
    print(addr)

    os.environ['MASTER_PORT'] = port
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend='nccl')

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size
