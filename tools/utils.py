import sys
import torch
import os
import shutil
from torch.utils.data.dataloader import DataLoader
sys.path.append('.')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)#torch.topk函数，对维度为1，从大到小排maxk个数。pred为对应的序号。
        pred = pred.t()#转置矩阵


        correct = pred.eq(target[None])#target[None]将target的tensor维数和ouput保持相同，维度不一定相同。

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)#前k列拉平求和。
            res.append(correct_k * (100.0 / batch_size))
        return res


def create_exp_dir(path, scripts_to_save=None):
    os.makedirs(path, exist_ok=True)

    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        script_path = os.path.join(path, 'scripts')
        if os.path.exists(script_path):
            shutil.rmtree(script_path)
        os.mkdir(script_path)
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            print(dst_file)
            shutil.copytree(script, dst_file)


class ForeverDataIterator:
    """A data iterator that will never stop producing data"""
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)#数据迭代器iter函数。

    def __next__(self):
        try:
            data = next(self.iter)#数据执行迭代。
        except StopIteration:#到最后一个数据，输出最后一个数据。停止迭代。
            self.iter = iter(self.data_loader)
            data = next(self.iter)#若是try异常，则执行except函数
        return data

    def __len__(self):
        return len(self.data_loader)


def get_datasets(root, tasks, transform, dataset):
    res = []
    for task in tasks.split("-"):
        res.append(dataset(root=root, task=task, download=True, transform=transform))
    datasets = torch.utils.data.ConcatDataset(res) 
    return datasets


