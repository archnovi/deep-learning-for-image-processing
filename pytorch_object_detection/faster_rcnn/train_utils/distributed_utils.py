from collections import defaultdict, deque
import datetime
import pickle
import time
import errno
import os

import torch
import torch.distributed as dist


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)  # deque简单理解成加强版list。可以在左边和右边加值。设置maxlen后是一个有界的list，如果往满编的list中再append元素，对面的元素会移除
        self.total = 0.0 # 记录所有值的和
        self.count = 0 # 目前看到了多少个值
        self.fmt = fmt  # 字符串输出的格式

    def update(self, value, n=1):  # 例如 value.update(acc1.item(), n=batch_size)  这样可以得到这个batch总和的acc1值(去除平均)，方便后面算全局平均。
        self.deque.append(value) # 将值加入列表中
        self.count += n # 更新count
        self.total += value * n  # 更新 total

    def synchronize_between_processes(self): # 用来在分布式训练时同步不同进程间的值
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized(): # 判断是不是在进行分布式训练，如果不是就直接返回
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda") # 将count，total转tensor来使用分布式训练
        dist.barrier()
        dist.all_reduce(t) # 让所有进程的count和total都得到最终结果
        t = t.tolist() # 转回list
        # 更新self里的count和total
        self.count = int(t[0])
        self.total = t[1]

    @property # @property 是装饰器，这里可简单理解为增加median属性(只读)
    def median(self):   # 计算中位数
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self): # 计算平均值
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):  # 计算总体平均数
        return self.total / self.count

    @property
    def max(self): # 计算最大值
        return max(self.deque)

    @property
    def value(self):  # 取最新的数值
        return self.deque[-1]

    def __str__(self):  # 用format格式化字符串输出
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return input_dict
    with torch.no_grad():  # 多GPU的情况
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size

        reduced_dict = {k: v for k, v in zip(names, values)}
        return reduced_dict


class MetricLogger(object): # 这个类不仅可以用来追踪所有的指标，自带平滑功能，而且还内置迭代器，能够完成计时等等功能。
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue) # defaultdict将默认字典内的value变为SmoothedValue对象
        self.delimiter = delimiter # 分隔符，这是配合字符串的join()方法使用的，默认为\t

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v) # 将meters内部对应的key更新其value。这里的updata是SmootheValue中的updata

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items(): # 以key-value的形式遍历meters字典
            # 按照"{}: {}".format(name, str(meter))的格式来生成字符串，例如loss: 0.6132
            loss_str.append(
                "{}: {}".format(name, str(meter)) 
            ) 
        return self.delimiter.join(loss_str) # 使用join()方法来生成一个用delimiter（/t）来分隔的字符串

    def synchronize_between_processes(self): # 读取所有meters字典里面的值(SmoothedValue)，并调用它们的synchronize_between_processes()方法来进行同步
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter): # 在字典中创建一个key为name，value为meter的key-value对，一般调用的时候meter传入SmoothedValue。
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header: # 如果没有传header参数，则默认为空字符串
            header = "" 
        start_time = time.time()
        end = time.time()
        # 创建两个SmoothedValue类来记录迭代时间和数据读取时间
        # 仅返回四位小数的平均值
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        # 创建空格分隔符，用来在打印的时候对齐
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        # 将要打印的信息放入列表，使用分隔符分隔这些信息
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([header,
                                           '[{0' + space_fmt + '}/{1}]', # 会被替换为目前所在的迭代次数与总迭代次数
                                           'eta: {eta}', # 剩余时间
                                           '{meters}', # 各种指标
                                           'time: {time}', # 迭代时间
                                           'data: {data}', # 数据读取时间
                                           'max mem: {memory:.0f}']) # 如果有cuda，那么列表多一项最大内存占用
        else:
            log_msg = self.delimiter.join([header,
                                           '[{0' + space_fmt + '}/{1}]',
                                           'eta: {eta}',
                                           '{meters}',
                                           'time: {time}',
                                           'data: {data}'])
        MB = 1024.0 * 1024.0 # 计算一个常数
        for obj in iterable:
            data_time.update(time.time() - end)  # 更新obj从iterable拿出来的时间
            yield obj  # 生成obj返回到调用的迭代器那里
            iter_time.update(time.time() - end)   # 更新外循环处理obj用的时间
            # 如果在打印频率上，或者是最后一个循环
            if i % print_freq == 0 or i == len(iterable) - 1: 
                eta_second = iter_time.global_avg * (len(iterable) - i)  # 估算剩余时间
                eta_string = str(datetime.timedelta(seconds=eta_second)) # 格式化剩余时间，将秒转为"h:m:s"
                if torch.cuda.is_available(): # 如果在用cuda训练
                    print(log_msg.format(i, len(iterable),
                                         eta=eta_string,
                                         meters=str(self),
                                         time=str(iter_time),
                                         data=str(data_time),
                                         memory=torch.cuda.max_memory_allocated() / MB))  # 打印显存占用
                else: # 不使用cuda就不打印显存占用
                    print(log_msg.format(i, len(iterable),
                                         eta=eta_string,
                                         meters=str(self),
                                         time=str(iter_time),
                                         data=str(data_time)))
            i += 1
            end = time.time() # 更新一个迭代完成后的时间点
        total_time = time.time() - start_time # 总花费时间是现在的时间减去最开始记录的时间
        total_time_str = str(datetime.timedelta(seconds=int(total_time))) # 秒转化为"h:m:s"
        # 格式化输出总时间，以及平均每个obj消耗的时间
        print('{} Total time: {} ({:.4f} s / it)'.format(header,
                                                         total_time_str,

                                                         total_time / len(iterable)))


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        """根据step数返回一个学习率倍率因子"""
        if x >= warmup_iters:  # 当迭代数大于给定的warmup_iters时，倍率因子为1
            return 1
        alpha = float(x) / warmup_iters
        # 迭代过程中倍率因子从warmup_factor -> 1
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    # 使用torch1.9或以上时建议加上device_ids=[args.rank]
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

