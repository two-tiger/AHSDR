import torch
from utils.status import progress_bar, create_stash
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from models.onlinevt import Onlinevt
from mydatasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from mydatasets import get_dataset
import sys
import copy
# from torchsummaryX import summary
# from ptflops import get_model_complexity_info
from torch import nn
import time
# from apex import amp
# from apex.parallel import DistributedDataParallel
# from apex.parallel import convert_syncbn_model
import torch.distributed as dist
# import megengine as mge
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from used_attacks import Logits_NI, PGD, MIFGSM, NIFGSM, TIFGSM, DIFGSM
import math
# import seaborn as sns

torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """

    if dataset.NAME == 'seq-core50':
        N_CLASSES_PER_TASK = [10, 5, 5, 5, 5, 5, 5, 5, 5]
        FROM_CLASS = int(np.sum(N_CLASSES_PER_TASK[:k]))
        TO_CLASS = int(np.sum(N_CLASSES_PER_TASK[:k+1]))
        outputs[:, 0:FROM_CLASS] = -float('inf')
        outputs[:, TO_CLASS: 50] = -float('inf')
    else:
        outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
        outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
                   dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset, task_id, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    gamma = 1
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):

        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            if len(data) == 3:
                inputs, labels, _ = data
            elif len(data) == 2:
                inputs, labels = data
            inputs, labels = inputs.to(model.device, non_blocking=True), labels.to(model.device, non_blocking=True)
            with torch.no_grad():
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                elif last:
                    outputs = model.old_means_pre(inputs)
                elif dataset.args.model == 'our':
                    outputs = model(inputs, dataset)
                else:
                    if (dataset.args.model == 'derppcct' or dataset.args.model == 'onlinevt') and model.net.net.distill_classifier:
                        outputs = model.net.net.distill_classification(inputs)
                        # outputs = model.ncm(inputs)
                        outputs[:, (task_id) * dataset.N_CLASSES_PER_TASK: (task_id) * dataset.N_CLASSES_PER_TASK+ dataset.N_CLASSES_PER_TASK] \
                            = outputs[:, (task_id) * dataset.N_CLASSES_PER_TASK: (task_id) * dataset.N_CLASSES_PER_TASK+ dataset.N_CLASSES_PER_TASK] * gamma
                    else:
                        outputs = model(inputs)

                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(round(correct / total * 100, 3)
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(round(correct_mask_classes / total * 100, 3))

    model.net.train(status)
    return accs, accs_mask_classes


def train(model: Onlinevt, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    model_stash = create_stash(model, args, dataset)
    results, results_mask_classes = [], []
    class_num_seen_img = {}
    for i in range(args.num_classes):
        class_num_seen_img[str(i)] = 0

    if args.csv_log:
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME, dataset.N_TASKS)

    start = time.time()
    print(file=sys.stderr)

    if hasattr(model.args, 'ce'):
        ce = model.args.ce
        model.args.ce = 1
    for t in range(dataset.N_TASKS):
        n_tasks = dataset.N_TASKS
        model.net.train() # backbone.utils.CCT_our.CVT(32, output_dim)
        if args.use_lr_scheduler:
            model.set_opt()
        train_loader, test_loader = dataset.get_data_loaders()
        if hasattr(model, 'begin_task'):
            if model.NAME != 'our':
                model.begin_task(dataset)
            elif args.begin_task:
                model.begin_task(dataset)
        if hasattr(model.args, 'ce'):
            if t > 0:
                model.args.ce = ce * model.args.ce
                pass
            print('model.args.ce: ', model.args.ce)
        for epoch in range(args.n_epochs - int(t*0)):

            model.display_img = False

            for i, data in enumerate(train_loader):
                if i == (train_loader.__len__()-1):
                    model.display_img = True
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.device, non_blocking=True)
                    labels = labels.to(model.device, non_blocking=True)
                    not_aug_inputs = not_aug_inputs.to(model.device, non_blocking=True)
                    logits = logits.to(model.device, non_blocking=True)
                    loss = model.observe(inputs, labels, not_aug_inputs, logits)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device, non_blocking=True), labels.to(
                        model.device, non_blocking=True)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    loss = model.observe(inputs, labels, not_aug_inputs,task_id = t)
                if args.use_inf:
                    model.buffer.find_low_decrease_inf(examples=not_aug_inputs, model=model.net.net,\
                            k = t, dataset = dataset, labels=labels)
                for label in labels:
                    class_num_seen_img[str(label.item())] += 1
                if args.find_in_step and args.use_perturbation:
                    if args.use_ema:
                        model.buffer.find_low_grad_trace_gaussian_perturbation_ema(inputs=inputs, not_aug_inputs=not_aug_inputs, labels=labels, model=model.net.net, class_num_seen_img=class_num_seen_img)
                    else:
                        model.buffer.find_low_grad_trace_gaussian_perturbation(inputs=inputs, not_aug_inputs=not_aug_inputs, labels=labels, model=model.net.net,class_num_seen_img=class_num_seen_img)
                    model.buffer.add_low_trace(n_tasks)
                elif args.find_in_step and args.use_attack:
                    if args.use_l2:
                        eps = math.sqrt(3*32*32*math.pow((args.eps/255),2))
                        alpha = math.sqrt(3*32*32*math.pow((args.alpha_atk/255),2))
                        l2 = True
                    else:
                        eps = (args.eps/255)
                        alpha = (args.alpha_atk/255)
                        l2 = False
                        
                    from typing import Union
                    buffer_attack_kwargs = dict(eps=eps, alpha=alpha, steps=args.steps, targeted=True, normalize=False,l2=l2)
                    attack:Union[PGD,MIFGSM,NIFGSM] = eval(args.buffer_attack)(model.net.net,**buffer_attack_kwargs)

                    if args.num_good is not None: #! add low and high loss imgs
                        adv_aug_inputs = attack(inputs, labels)
                        model.buffer.find_low_high_decrease(examples=not_aug_inputs, adv_aug_inputs=adv_aug_inputs,\
                                                        model=model.net.net, labels=labels)
                        model.buffer.add_low_high_decrease(n_tasks,args.add_adv,task=t,\
                                                        percent=args.num_good,folder=args.out_dir)
                    else: #! add low loss decrease imgs
                        adv_aug_inputs = attack(inputs, labels)
                        if args.use_ema:
                            model.buffer.find_low_decrease_ema(examples=inputs, not_aug_inputs = not_aug_inputs, adv_aug_inputs=adv_aug_inputs,\
                                                        model=model.net.net, labels=labels,lamda_grad_norm=args.lamda_grad_norm)
                        else:
                            model.buffer.find_low_decrease(examples=inputs, not_aug_inputs = not_aug_inputs, adv_aug_inputs=adv_aug_inputs,\
                                                        model=model.net.net, labels=labels,lamda_grad_norm=args.lamda_grad_norm)
                        model.buffer.add_low_decrease(n_tasks,args.add_adv,task=t,folder=args.out_dir,save_img=args.save_img,transform_img=inputs)
                    
                # progress_bar(i, len(train_loader), epoch, t, loss)

                if hasattr(model, 'middle_task') and (i % 2000) == 0 and i > 0 and dataset.NAME == 'seq-mnist':
                    tmp_buffer = copy.deepcopy(model.buffer)
                    model.middle_task(dataset)
                    accs = evaluate(model, dataset, t)
                    model.buffer = tmp_buffer
                    print(accs)
                    mean_acc = np.mean(accs, axis=1)
                    print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

                model_stash['batch_idx'] = i + 1
            model_stash['epoch_idx'] = epoch + 1
            model_stash['batch_idx'] = 0
            if model._scheduler is not None:
                model._scheduler.step()
            if args.use_inf:
                model.buffer.add_low_decrease_inf(n_tasks,task=t,folder=args.out_dir)
        
        model_stash['task_idx'] = t + 1
        model_stash['epoch_idx'] = 0

        if hasattr(model, 'end_task'):
            if model.NAME == 'our':
                model.end_task(dataset, t)
            else:
                model.end_task(dataset)

        accs = evaluate(model, dataset, t)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)


        if args.use_attack and t < dataset.N_TASKS-1 and not args.find_in_step: #! attack method
            if args.use_l2:
                eps = math.sqrt(3*32*32*math.pow((args.eps/255),2))
                alpha = math.sqrt(3*32*32*math.pow((args.alpha_atk/255),2))
                l2 = True
            else:
                eps = (args.eps/255)
                alpha = (args.alpha_atk/255)
                l2 = False
            if args.buffer_attack == 'PGD':
                attack = PGD(model.net.net, eps=eps, alpha=alpha, steps=args.steps, targeted=True, normalize=False,l2=l2)
            elif args.buffer_attack == 'MIFGSM':
                attack = MIFGSM(model.net.net, eps=eps, alpha=alpha, steps=args.steps, targeted=True, normalize=False,l2=l2)
            elif args.buffer_attack == 'NIFGSM':
                attack = NIFGSM(model.net.net, eps=eps, alpha=alpha, steps=args.steps, targeted=True, normalize=False,l2=l2)

            if args.num_good is not None: #! add low and high loss imgs
                for data in train_loader:
                    inputs, labels, not_aug_inputs = data
                    adv_aug_inputs = attack(inputs, labels)
                    model.buffer.find_low_high_decrease(examples=not_aug_inputs, adv_aug_inputs=adv_aug_inputs,\
                                                    model=model.net.net, labels=labels)
                model.buffer.add_low_high_decrease(n_tasks,args.add_adv,task=t,\
                                                percent=args.num_good,folder=args.out_dir)
            else: #! add low loss decrease imgs
                for data in train_loader:
                    inputs, labels, not_aug_inputs = data
                    adv_aug_inputs = attack(inputs, labels)
                    if args.use_ema:
                        model.buffer.find_low_decrease_ema(examples=inputs, not_aug_inputs = not_aug_inputs, adv_aug_inputs=adv_aug_inputs,\
                                                    model=model.net.net, labels=labels,lamda_grad_norm=args.lamda_grad_norm)
                    else:
                        model.buffer.find_low_decrease(examples=inputs, not_aug_inputs = not_aug_inputs, adv_aug_inputs=adv_aug_inputs,\
                                                    model=model.net.net, labels=labels,lamda_grad_norm=args.lamda_grad_norm)
                model.buffer.add_low_decrease(n_tasks,args.add_adv,task=t,folder=args.out_dir,save_img=args.save_img,transform_img=inputs)
                
        elif args.use_perturbation and t < dataset.N_TASKS-1 and not args.find_in_step:
            for data in train_loader:
                inputs, labels, not_aug_inputs = data
                if args.use_ema:
                    model.buffer.find_low_grad_trace_gaussian_perturbation_ema(inputs=inputs, not_aug_inputs=not_aug_inputs, labels=labels, model=model.net.net)
                else:
                    model.buffer.find_low_grad_trace_gaussian_perturbation(inputs=inputs, not_aug_inputs=not_aug_inputs, labels=labels, model=model.net.net)
            model.buffer.add_low_trace(n_tasks)
            
        else:
            pass
        
        model_stash['mean_accs'].append(mean_acc)

        if args.csv_log:
            csv_logger.log(mean_acc)
            csv_logger.log_class_detail(results)
            csv_logger.log_task_detail(results_mask_classes)

        if hasattr(model.net, 'frozen'):
            model.net.frozen(t)

    end = time.time()
    time_train = round(end - start, 1)
    print('running time: ', time_train, ' s')
    if args.csv_log:
        csv_logger.log_time(time_train)
        csv_logger.add_bwt(results, results_mask_classes)
        csv_logger.add_forgetting(results, results_mask_classes)
        csv_logger.write(vars(args))
    print(f'forgetting: {csv_logger.forgetting}')
    print(f'forgetting_mask_classes: {csv_logger.forgetting_mask_classes}')
