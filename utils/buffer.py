import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from torchvision import transforms
import time
import torchvision
import os
import math
# from hessian import hutchinson_trace_hvp
from used_attacks import MIFGSM
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import grad


def compute_p_gradnorm(model, x, labels, k, dataset):
    cri = nn.CrossEntropyLoss(reduction='none')
    x.requires_grad = True
    logit = model(x)
    # mask = torch.zeros_like(logit)
    # mask[:, k * dataset.N_CLASSES_PER_TASK:(k + 1) * dataset.N_CLASSES_PER_TASK] = 1
    logit = logit
    probs = torch.softmax(logit, dim=-1)
    # max_indices = torch.argmax(logit, dim=1)
    # second_max_indices = torch.argsort(logit, dim=1)[:, -2]
    # max_values = torch.gather(logit, 1, labels.view(-1, 1))
    # second_max_values = torch.gather(logit, 1, second_max_indices.unsqueeze(1))
    # on_board = torch.gather(probs, 1, labels.view(-1, 1))
    loss = cri(logit, labels)
    grads_ori = grad(loss, x, grad_outputs=torch.ones_like(loss), create_graph=True)[0]
    # x_new = x + 0.01 * torch.randn_like(x)
    # x_new = x_new.requires_grad_()
    # logit_new = model(x_new)
    # loss_new = cri(logit_new * mask, labels)
    # grads_new = grad(grads_ori, x, grad_outputs=torch.ones_like(grads_ori), create_graph=True)[0]
    # print(grads_new.shape)
    # print(torch.gather(probs, 1, labels.view(-1, 1)).shape, torch.norm(grads.view(x.shape[0], -1), dim=-1).shape)
    loss = loss + torch.norm(grads_ori.view(x.shape[0], -1), dim=-1)
    # loss = -(1-torch.gather(probs, 1, labels.view(-1, 1))).squeeze(1) * torch.norm(grads_ori.view(x.shape[0], -1), dim=-1)#torch.norm(grads_new.view(x.shape[0], -1), dim=-1)
    
    # loss = max_values - second_max_values
    # loss[loss<=0] = float('inf')
    # loss = torch.randn_like(loss)
    # print(loss.shape)
    return loss

def grad_loss(model, x, labels,lamda_grad_norm=0.1):
    cri = nn.CrossEntropyLoss(reduction='none')
    x = x.requires_grad_()
    logit = model(x)
    loss = cri(logit, labels)
    grads = grad(loss, x, grad_outputs=torch.ones_like(loss), create_graph=True)[0]
    loss = loss + lamda_grad_norm * torch.norm(grads.view(x.shape[0], -1), dim=-1)
    return loss

def compute_softmax_gradient(model, x, k, dataset):
    """
    Compute the gradient of logsoftmax probabilities with respect to the input.

    Parameters:
    - model: PyTorch model
    - x: Input tensor
    - y: True labels

    Returns:
    - gradient: Gradient of logsoftmax probabilities with respect to input, flattened
    """

    B, C, H, W = x.shape # Enable gradient computation for input
    # print(x.shape)
    # # Forward pass
    logits = model(x)
    mask = torch.zeros_like(logits)
    mask[:, :k * dataset.N_CLASSES_PER_TASK] = 1
    logits = logits * mask
    # L = logits.shape[1]
    # # Calculate logsoftmax probabilities
    softmax_probs = torch.nn.functional.softmax(logits, dim=1)
    # logsoftmax_probs = torch.log(softmax_probs)
    # Compute gradient of logsoftmax probabilities with respect to input
    grad_logsoftmax = torch.zeros(x.shape[0], logits.shape[1], *x.shape[1:]).to(x.device)
    # grad_logsoftmax = torch.zeros(x.shape[0], logits.shape[1], C * H * W).to(x.device)
    # grad_logsoftmax = grad(logsoftmax_probs, x, grad_outputs=torch.ones_like(logsoftmax_probs))[0]
    for i in range(x.shape[0]):
        # 使用 grad 计算梯度，同时传递 grad_outputs 以创建梯度图
        input = x[i].unsqueeze(0).requires_grad_()
        logit = model(input)
        softmax_prob = torch.nn.functional.softmax(logit, dim=1)
        logsoftmax_prob = torch.log(softmax_prob)
        grad_logsoftmax[i] = grad(logsoftmax_prob, input, grad_outputs=torch.ones_like(logsoftmax_prob), create_graph=True)[0]
    # Flatten the gradient
    # gradient = grad_logsoftmax.view(x.shape[0], -1)
    # print(softmax_probs.shape, torch.norm(grad_logsoftmax.view(B, -1, C * H * W), dim=2).shape)
    
    inf_mat = grad_logsoftmax.view(B, -1, C * H * W)
    inf = softmax_probs[:, :, None, None] * torch.einsum('blc,bld->bldc', inf_mat, inf_mat)
    
    inf[:, :k * dataset.N_CLASSES_PER_TASK] = 0
    inf[:, (k + 1) * dataset.N_CLASSES_PER_TASK:] = 0
    # inf = inf
    inf = -inf.sum(dim=1)
    inf = torch.sum(torch.diagonal(inf, dim1=-2, dim2=-1), dim=-1)
    # inf = softmax_probs * torch.norm(grad_logsoftmax.view(B, -1, C * H * W), dim=2)
    # print(inf.shape)
    # mask = torch.zeros_like(inf)
    # mask[:, :(k + 1) * dataset.N_CLASSES_PER_TASK] = 1
    # inf = -torch.sum(inf , dim = -1) 
    # print(inf.shape)
    # print(inf.item())
    return inf

def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class Buffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, device, args=None, n_tasks=None, mode='reservoir'):
        assert mode in ['ring', 'reservoir']
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        self.args = args
        self.task_now = 0
        self.num_classes = args.num_classes

        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.class_decreased_inf = {}
        self.class_inf_min_images = {}
        self.class_decreased_loss = {}
        self.class_decreased_loss_ema = {}
        self.class_low_trace = {}
        self.class_low_trace_ema = {}
        self.class_images = {}
        self.class_adv_images = {}
        self.class_decreased_most_loss = {}
        self.class_most_images = {}
        self.class_most_adv_images = {}
        # 初始化每类下降的损失存储器
        for i in range(self.num_classes):
            self.class_decreased_loss[str(i)] = []
            self.class_decreased_loss_ema[str(i)] = 0
            self.class_low_trace[str(i)] = []
            self.class_low_trace_ema[str(i)] = 0
        # 初始化每类的原图片
        for i in range(self.num_classes):
            self.class_images[str(i)] = []
            # 初始化每类的攻击后图片
            self.class_adv_images[str(i)] = []
            # 初始化每类下降的最多损失存储器
            self.class_decreased_most_loss[str(i)] = []
            # 初始化每类下降最多的原图片
            self.class_most_images[str(i)] = []
            # 初始化每类下降最多的攻击后图片
            self.class_most_adv_images[str(i)] = []
            self.class_decreased_inf[str(i)] = []
        self.current_task_labels = []
        self.num_seen_class_examples = {}
        for i in range(self.num_classes):
            self.num_seen_class_examples[str(i)] = 0

        print(n_tasks)
        if mode == 'ring':

            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['examples', 'labels', 'logits', 'task_labels', 'features']
        # self.examples = None
        if hasattr(args, 'dataset') and 'imagenet' in args.dataset:
            self.transform_type_A = False
        else:
            self.transform_type_A = False

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor, features: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :param features: tensor containing the latent features
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device))

    def add_data(self, examples, labels=None, logits=None, task_labels=None, features=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :param features: tensor containing the latent features
        :return:
        """
        # if self.buffer_size == 20:
        #     time.sleep(100)
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels, features)

        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)
                if features is not None:
                    self.features[index] = task_labels[i].to(self.device)
            # if index >= 0:
            #     self.examples[index] = examples[i]
            #     if labels is not None:
            #         self.labels[index] = labels[i]
            #     if logits is not None:
            #         self.logits[index] = logits[i]
            #     if task_labels is not None:
            #         self.task_labels[index] = task_labels[i]
            #     if features is not None:
            #         self.features[index] = task_labels[i]

    def add_data_our(self, examples, labels=None, logits=None, task_labels=None, features=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :param features: tensor containing the latent features
        :return:
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels, features)

        for i in range(examples.shape[0]):

            self.examples[self.num_seen_examples] = examples[i].to(self.device)
            if labels is not None:
                self.labels[self.num_seen_examples] = labels[i].to(self.device)
            if logits is not None:
                self.logits[self.num_seen_examples] = logits[i].to(self.device)
            if task_labels is not None:
                self.task_labels[self.num_seen_examples] = task_labels[i].to(self.device)
            if features is not None:
                self.features[self.num_seen_examples] = features[i].to(self.device)

            self.num_seen_examples += 1

    def get_data(self, size: int, transform: transforms=None,iter_num=None,print_freq=None,task_id=None) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]), size=size, replace=False)
        
        # if iter_num is not None and print_freq is not None and iter_num % print_freq == 0:
        #     print(f'num_seen_examples:{self.num_seen_examples}')
        #     filename = os.path.join(self.args.out_dir, 'task-{}-got.png'.format(task_id))
        #     selected = torch.stack([ee for ee in self.examples[choice]])
        #     torchvision.utils.save_image(selected, filename, nrow=selected.shape[0], normalize=False)
        
        if transform is None: transform = lambda x: x
        if self.transform_type_A:
            ret_tuple = (torch.stack([transform(image=ee.cpu().numpy())['image'] for ee in self.examples[choice]]).to(self.device),)
        else:
            ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).to(self.device),)
            aug_2_view = (torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)
        if hasattr(self.args, 'model') and self.args.model == 'onlinevt':
            ret_tuple += aug_2_view
        return ret_tuple

    def get_all_data(self, transform: transforms=None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        # if hasattr(self, 'examples'):

        if transform is None: transform = lambda x: x
        # ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples]).to(self.device),)
        if self.transform_type_A and transform is None:
            ret_tuple = (torch.stack([transform(image=ee.cpu().numpy())['image'] for ee in self.examples]).to(self.device),)
        else:
            ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)

        return ret_tuple

    def get_all_data_domain(self, transform: transforms=None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        # if hasattr(self, 'examples'):

        if transform is None: transform = lambda x: x
        if self.transform_type_A:
            ret_tuple = (torch.stack([transform(image=ee.cpu().numpy())['image']
                                for ee in self.examples]).to(self.device),)
        else:
            ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)

        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0

    def get_data_labels(self, transform: transforms = None) -> Tuple:
            """
            Return all the items in the memory buffer.
            :param transform: the transformation to be applied (data augmentation)
            :return: a tuple with all the items in the memory buffer
            """
            if transform is None:
                transform = lambda x: x
            # print(len(self.adv_idx))
            print(self.adv_idx)
            # ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples]).to(self.device),)
            ret_samples = torch.stack([transform(ee.cpu()) for ee in self.examples[self.adv_idx]]).to(
                    self.device
                )
            ret_labels = self.labels[self.adv_idx].to(
                    self.device
                )
            print(ret_labels)
            return ret_samples,ret_labels
        
    def put_data(self, adv_aug_inputs,model) -> None:
        """
        Return images back to thier place in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        loss = nn.CrossEntropyLoss(reduction='none')
        labels = self.labels[self.adv_idx].clone().to(self.device)
        ori_images = torch.stack([ee for ee in self.examples[self.adv_idx]]).to(self.device)

        ori_outputs = model(ori_images)
        adv_outputs = model(adv_aug_inputs)

        ori_loss = loss(ori_outputs,labels)
        adv_loss = loss(adv_outputs,labels)
        decreased_loss = ori_loss - adv_loss
        decreased_percent = decreased_loss / ori_loss
        print(f'the original loss is: {ori_loss}\n')
        print(f'the adv loss is: {adv_loss}\n')
        print(f'the decreased loss is: {decreased_loss}\n')
        print(f'the decreased percent is: {decreased_percent}\n')

        for i in range(len(self.adv_idx)):
            adv_idx = self.adv_idx[i]
            self.examples[adv_idx] = adv_aug_inputs[i].to(self.device)
        self.adv_idx = []

    def find_low_decrease(
        self, examples, not_aug_inputs, adv_aug_inputs, model, labels=None,lamda_grad_norm=0.1
    ):
        """
        find the data to the memory buffer according to the lowest decreased loss.
        :param examples: tensor containing the original images
        :param adv_aug_inputs: tensor containing the adv images
        :param model: the model net to calculate loss
        :param labels: tensor containing the labels
        :return:
        """
        max_class_buffer_size = self.buffer_size // self.num_classes
        if not hasattr(self, "examples"):
            self.init_tensors(not_aug_inputs, labels,None,None,None)

        #! 批量计算每一张图片攻击前后下降的loss
        loss = self.loss
        ori_images = examples.to(self.device)
        labels = labels.to(self.device)
        # ori_outputs = model(ori_images)
        adv_outputs = model(adv_aug_inputs)

        # ori_loss = loss(ori_outputs,labels)
        adv_loss = loss(adv_outputs,labels)

        ori_loss = grad_loss(model, ori_images, labels,lamda_grad_norm) 
        # adv_loss = grad_loss(model, adv_aug_inputs, labels)

        decreased_loss = ori_loss - adv_loss#torch.abs(ori_outputs - adv_outputs).sum(dim=-1) + (ori_loss - adv_loss)
        # decreased_loss = (grad_loss(model, ori_images, labels) - ori_loss) * (ori_loss - adv_loss)
        for i in range(examples.shape[0]):
            label_int = int(labels[i].item())
            if label_int not in self.current_task_labels:
                self.current_task_labels.append(label_int)
            label = str(labels[i].item())
            #! 不到每类的最大个数，就直接把下降的loss和图片加进list中
            if len(self.class_decreased_loss[label]) < max_class_buffer_size:
                self.class_decreased_loss[label].append(decreased_loss[i].item())
                self.class_images[label].append(not_aug_inputs[i])
                self.class_adv_images[label].append(adv_aug_inputs[i])
            #! 达到每类最大个数后，则进行判断
            else:
                #! 若当前下降的损失小于list中的最大值，则将当前图片和损失替换进list
                if decreased_loss[i].item() < max(self.class_decreased_loss[label]):
                    idx = self.class_decreased_loss[label].index(max(self.class_decreased_loss[label]))
                    self.class_decreased_loss[label][idx] = decreased_loss[i].item()
                    self.class_images[label][idx] = not_aug_inputs[i]
                    self.class_adv_images[label][idx] = adv_aug_inputs[i]

    def add_low_decrease(self, n_tasks, add_adv:bool, task, folder,save_img,transform_img):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :param features: tensor containing the latent features
        :return:
        """
        max_class_buffer_size = self.buffer_size // self.num_classes
        #! 更新buffer_count
        self.num_seen_examples += self.buffer_size // n_tasks
        not_aug_inputs = []
        adv_aug_inputs = []
        for i in range(len(self.current_task_labels)):
            label_int = self.current_task_labels[i]
            label = str(label_int)

            for j in range(label_int * max_class_buffer_size,\
                        (label_int+1) * max_class_buffer_size):
                if add_adv:
                    self.examples[j] = self.class_adv_images[label][j % max_class_buffer_size]
                else:
                    self.examples[j] = self.class_images[label][j % max_class_buffer_size]
                self.labels[j] = torch.tensor(label_int).to(self.device)
        
        #! 保存该task图片，可视化
        min_label = min(self.current_task_labels)
        max_label = max(self.current_task_labels)
        for i in range(min_label,max_label+1):
            if add_adv:
                for j in range(max_class_buffer_size):
                    adv_aug_inputs.append(self.examples[i*max_class_buffer_size+j])
                for img in self.class_images[str(i)]:
                    not_aug_inputs.append(img)
                # for img in self.class_adv_images[str(i)]:
                #     adv_aug_inputs.append(img)
            else:
                for j in range(max_class_buffer_size):
                    not_aug_inputs.append(self.examples[i*max_class_buffer_size+j])
                # for img in self.class_images[str(i)]:
                #     not_aug_inputs.append(img)
                for img in self.class_adv_images[str(i)]:
                    adv_aug_inputs.append(img)
        not_aug_inputs = torch.stack(not_aug_inputs)
        adv_aug_inputs = torch.stack(adv_aug_inputs)
        # print('1')
        if save_img:
            save_images(transform_img, adv_aug_inputs, task, folder)
        #! 一个task结束后清空缓存器
        self.current_task_labels = []
        
    def find_low_decrease_ema(
        self, examples, not_aug_inputs, adv_aug_inputs, model, labels=None,lamda_grad_norm=0.1
    ):
        """
        find the data to the memory buffer according to the lowest decreased loss.
        :param examples: tensor containing the original images
        :param adv_aug_inputs: tensor containing the adv images
        :param model: the model net to calculate loss
        :param labels: tensor containing the labels
        :return:
        """
        max_class_buffer_size = self.buffer_size // self.num_classes
        if not hasattr(self, "examples"):
            self.init_tensors(not_aug_inputs, labels,None,None,None)

        #! 批量计算每一张图片攻击前后下降的loss
        loss = self.loss
        ori_images = examples.to(self.device)
        labels = labels.to(self.device)
        # ori_outputs = model(ori_images)
        adv_outputs = model(adv_aug_inputs)

        # ori_loss = loss(ori_outputs,labels)
        adv_loss = loss(adv_outputs,labels)

        ori_loss = grad_loss(model, ori_images, labels,lamda_grad_norm) 
        # adv_loss = grad_loss(model, adv_aug_inputs, labels)

        decreased_loss = ori_loss - adv_loss#torch.abs(ori_outputs - adv_outputs).sum(dim=-1) + (ori_loss - adv_loss)
        # decreased_loss = (grad_loss(model, ori_images, labels) - ori_loss) * (ori_loss - adv_loss)
        
        for i in range(examples.shape[0]):
            label_int = int(labels[i].item())
            if label_int not in self.current_task_labels:
                self.current_task_labels.append(label_int)
            label = str(labels[i].item())
            delta_decreased_loss = decreased_loss[i] - self.class_decreased_loss_ema[label]
            self.class_decreased_loss_ema[label] = 0.8 * self.class_decreased_loss_ema[label] + (1 - 0.8) * decreased_loss[i].detach()
            #! 不到每类的最大个数，就直接把下降的loss和图片加进list中
            if len(self.class_decreased_loss[label]) < max_class_buffer_size:
                self.class_decreased_loss[label].append(delta_decreased_loss.item())
                self.class_images[label].append(not_aug_inputs[i])
                self.class_adv_images[label].append(adv_aug_inputs[i])
            #! 达到每类最大个数后，则进行判断
            else:
                #! 若当前下降的损失小于list中的最大值，则将当前图片和损失替换进list
                if delta_decreased_loss.item() < max(self.class_decreased_loss[label]):
                    idx = self.class_decreased_loss[label].index(max(self.class_decreased_loss[label]))
                    self.class_decreased_loss[label][idx] = delta_decreased_loss.item()
                    self.class_images[label][idx] = not_aug_inputs[i]
                    self.class_adv_images[label][idx] = adv_aug_inputs[i]
                    
    def find_low_grad_trace_gaussian_perturbation(self, inputs, not_aug_inputs, labels, model):
        max_class_buffer_size = self.buffer_size // self.num_classes
        grad_list = []
        labels = labels.to(self.device)
        # 生成3个高斯扰动样本
        for _ in range(3):
            cri = nn.CrossEntropyLoss(reduction='none')
            noise = torch.rand_like(inputs) * 0.1
            perturbed_inputs = inputs + noise
            perturbed_inputs = perturbed_inputs.to(self.device)
            perturbed_inputs.requires_grad = True
            logit = model(perturbed_inputs)
            loss = cri(logit, labels)
            grads = grad(loss, perturbed_inputs, grad_outputs=torch.ones_like(loss), create_graph=True)[0].view(inputs.shape[0], -1)
            grad_list.append(grads)
            
        # 将梯度堆叠为 (3, batch_size, num_features)
        grad_matrix = torch.stack(grad_list)
        
        # 计算梯度协方差矩阵
        mean_grad = torch.mean(grad_matrix, dim=0, keepdim=True)
        centered_grads = grad_matrix - mean_grad
        cov_matrix = torch.matmul(centered_grads.permute(1, 2, 0), centered_grads.permute(1, 0, 2)) / (grad_matrix.shape[0] - 1)
        
        # 计算协方差矩阵的迹
        trace_cov = torch.einsum("bii->b", cov_matrix)
        
        # for i in range(inputs.shape[0]):
        #     label_int = int(labels[i].item())
        #     if label_int not in self.current_task_labels:
        #         self.current_task_labels.append(label_int)
        #     label = str(labels[i].item())
        #     #! 不到每类的最大个数，就直接把迹和图片加进list中
        #     if len(self.class_low_trace[label]) < max_class_buffer_size:
        #         self.class_low_trace[label].append(trace_cov[i].item())
        #         self.class_images[label].append(not_aug_inputs[i])
        #     #! 达到每类最大个数后，则进行判断
        #     else:
        #         #! 若当前的迹小于list中的最大值，则将当前图片和迹替换进list
        #         if trace_cov[i].item() < max(self.class_low_trace[label]):
        #             idx = self.class_low_trace[label].index(max(self.class_low_trace[label]))
        #             self.class_low_trace[label][idx] = trace_cov[i].item()
        #             self.class_images[label][idx] = not_aug_inputs[i]
                    
        for i in range(inputs.shape[0]):
            label_int = int(labels[i].item())
            if label_int not in self.current_task_labels:
                self.current_task_labels.append(label_int)
            label = str(labels[i].item())
            #! 不到每类的最大个数，就直接把迹和图片加进list中
            if len(self.class_low_trace[label]) < max_class_buffer_size:
                self.class_low_trace[label].append(trace_cov[i].item())
                self.class_images[label].append(not_aug_inputs[i])
            #! 达到每类最大个数后，则进行判断
            else:
                #! 若当前的迹小于list中的最大值，则将当前图片和迹替换进list
                if trace_cov[i].item() > min(self.class_low_trace[label]):
                    idx = self.class_low_trace[label].index(min(self.class_low_trace[label]))
                    self.class_low_trace[label][idx] = trace_cov[i].item()
                    self.class_images[label][idx] = not_aug_inputs[i]
                    
    def find_low_grad_trace_gaussian_perturbation_ema(self, inputs, not_aug_inputs, labels, model):
        max_class_buffer_size = self.buffer_size // self.num_classes
        grad_list = []
        labels = labels.to(self.device)
        # 生成3个高斯扰动样本
        for _ in range(3):
            cri = nn.CrossEntropyLoss(reduction='none')
            noise = torch.rand_like(inputs) * 0.1
            perturbed_inputs = inputs + noise
            perturbed_inputs = perturbed_inputs.to(self.device)
            perturbed_inputs.requires_grad = True
            logit = model(perturbed_inputs)
            loss = cri(logit, labels)
            grads = grad(loss, perturbed_inputs, grad_outputs=torch.ones_like(loss), create_graph=True)[0].view(inputs.shape[0], -1)
            grad_list.append(grads)
            
        # 将梯度堆叠为 (3, batch_size, num_features)
        grad_matrix = torch.stack(grad_list)
        
        # 计算梯度协方差矩阵
        mean_grad = torch.mean(grad_matrix, dim=0, keepdim=True)
        centered_grads = grad_matrix - mean_grad
        cov_matrix = torch.matmul(centered_grads.permute(1, 2, 0), centered_grads.permute(1, 0, 2)) / (grad_matrix.shape[0] - 1)
        
        # 计算协方差矩阵的迹
        trace_cov = torch.einsum("bii->b", cov_matrix)
        
        for i in range(inputs.shape[0]):
            label_int = int(labels[i].item())
            if label_int not in self.current_task_labels:
                self.current_task_labels.append(label_int)
            label = str(labels[i].item())
            delta_trace = trace_cov[i] - self.class_low_trace_ema[label]
            self.class_low_trace_ema[label] = 0.8 * self.class_low_trace_ema[label] + (1 - 0.8) * trace_cov[i].detach()
            #! 不到每类的最大个数，就直接把迹和图片加进list中
            if len(self.class_low_trace[label]) < max_class_buffer_size:
                self.class_low_trace[label].append(delta_trace.item())
                self.class_images[label].append(not_aug_inputs[i])
            #! 达到每类最大个数后，则进行判断
            else:
                #! 若当前的迹小于list中的最大值，则将当前图片和迹替换进list
                if delta_trace.item() < max(self.class_low_trace[label]):
                    idx = self.class_low_trace[label].index(max(self.class_low_trace[label]))
                    self.class_low_trace[label][idx] = delta_trace.item()
                    self.class_images[label][idx] = not_aug_inputs[i]
                    
    def add_low_trace(self, n_tasks):
        max_class_buffer_size = self.buffer_size // self.num_classes
        #! 更新buffer_count
        self.num_seen_examples += self.buffer_size // n_tasks
        not_aug_inputs = []
        for i in range(len(self.current_task_labels)):
            label_int = self.current_task_labels[i]
            label = str(label_int)

            class_buffer_num = min(max_class_buffer_size, len(self.class_images[label]))
            for j in range(label_int * max_class_buffer_size, label_int * max_class_buffer_size + class_buffer_num):
                self.examples[j] = self.class_images[label][j % max_class_buffer_size]
                self.labels[j] = torch.tensor(label_int).to(self.device)
        #! 一个task结束后清空缓存器
        self.current_task_labels = []

    def update_current_task_labels(self, labels):
        # 获取当前任务中不存在的标签
        new_labels = set(labels.tolist()) - set(self.current_task_labels)

        # 将新标签加入当前任务标签列表
        self.current_task_labels.extend(new_labels)

    def find_low_decrease_inf(
        self, examples, model, k, dataset, labels=None
    ):
        """
        find the data to the memory buffer according to the lowest decreased loss.
        :param examples: tensor containing the original images
        :param adv_aug_inputs: tensor containing the adv images
        :param model: the model net to calculate loss
        :param labels: tensor containing the labels
        :return:
        """
        max_class_buffer_size = self.buffer_size // self.num_classes
        if not hasattr(self, "examples"):
            self.init_tensors(examples, labels,None,None,None)

        #! 批量计算每一张图片攻击前后下降的loss
        ori_images = examples.to(self.device)
        labels = labels.to(self.device)
        infs = compute_softmax_gradient(model, ori_images, k, dataset)
        # infs = compute_p_gradnorm(model, ori_images, labels, k, dataset)
        # self.update_current_task_labels(labels)
        # print(len(self.class_decreased_inf['0']))
        for i in range(examples.shape[0]):
            label_int = int(labels[i].item())
            if label_int not in self.current_task_labels:
                self.current_task_labels.append(label_int)
            label = str(labels[i].item())
            #! 不到每类的最大个数，就直接把下降的loss和图片加进list中
            if len(self.class_decreased_inf[label]) < max_class_buffer_size:
                self.class_decreased_inf[label].append(infs[i].item())
                self.class_images[label].append(examples[i])
            #! 达到每类最大个数后，则进行判断
            else:
                #! 若当前下降的损失小于list中的最大值，则将当前图片和损失替换进list
                if infs[i].item() < max(self.class_decreased_inf[label]):
                    idx = self.class_decreased_inf[label].index(max(self.class_decreased_inf[label]))
                    self.class_decreased_inf[label][idx] = infs[i].item()
                    self.class_images[label][idx] = examples[i]

    def add_low_decrease_inf(self, n_tasks, task, folder):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :param features: tensor containing the latent features
        :return:
        """
        max_class_buffer_size = self.buffer_size // self.num_classes
        #! 更新buffer_count
        self.num_seen_examples += self.buffer_size // n_tasks
        not_aug_inputs = []
        for i in range(len(self.current_task_labels)):
            label_int = self.current_task_labels[i]
            label = str(label_int)
            # print(label, self.current_task_labels, len(self.class_images[label]))
            for j in range(label_int * max_class_buffer_size,\
                        (label_int+1) * max_class_buffer_size):
                # print(j % max_class_buffer_size)
                self.examples[j] = self.class_images[label][j % max_class_buffer_size]
                self.labels[j] = torch.tensor(label_int).to(self.device)
        
        #! 保存该task图片，可视化
        min_label = min(self.current_task_labels)
        max_label = max(self.current_task_labels)
        for i in range(min_label,max_label+1):
            for j in range(max_class_buffer_size):
                not_aug_inputs.append(self.examples[i*max_class_buffer_size+j])
                # for img in self.class_images[str(i)]:
                #     not_aug_inputs.append(img)
        not_aug_inputs = torch.stack(not_aug_inputs)
        # save_images(not_aug_inputs, adv_aug_inputs, task, folder)
        #! 一个task结束后清空缓存器
        if self.task_now != task:
            self.current_task_labels = []
            self.task_now = task

    def find_low_high_decrease(
        self, examples, adv_aug_inputs, model, labels=None
    ):
        """
        find the data to the memory buffer according to the lowest and highest decreased loss.
        :param examples: tensor containing the original images
        :param adv_aug_inputs: tensor containing the adv images
        :param model: the model net to calculate loss
        :param labels: tensor containing the labels
        :return:
        """
        max_class_buffer_size = self.buffer_size // self.num_classes
        if not hasattr(self, "examples"):
            self.init_tensors(examples, labels,None,None,None)

        #! 批量计算每一张图片攻击前后下降的loss
        loss = self.loss
        ori_images = examples.to(self.device)
        labels = labels.to(self.device)
        ori_outputs = model(ori_images)
        adv_outputs = model(adv_aug_inputs)

        ori_loss = loss(ori_outputs,labels)
        adv_loss = loss(adv_outputs,labels)
        decreased_loss = ori_loss - adv_loss

        for i in range(examples.shape[0]):
            label_int = int(labels[i].item())
            if label_int not in self.current_task_labels:
                self.current_task_labels.append(label_int)
            label = str(labels[i].item())
            #! 不到每类的最大个数，就直接把下降的loss和图片加进list中
            if len(self.class_decreased_loss[label]) < max_class_buffer_size:
                self.class_decreased_loss[label].append(decreased_loss[i].item())
                self.class_images[label].append(examples[i])
                self.class_adv_images[label].append(adv_aug_inputs[i])
            #! 达到每类最大个数后，则进行判断
            else:
                #! 若当前下降的损失小于list中的最大值，则将当前图片和损失替换进list
                if decreased_loss[i].item() < max(self.class_decreased_loss[label]):
                    idx = self.class_decreased_loss[label].index(max(self.class_decreased_loss[label]))
                    self.class_decreased_loss[label][idx] = decreased_loss[i].item()
                    self.class_images[label][idx] = examples[i]
                    self.class_adv_images[label][idx] = adv_aug_inputs[i]
            # loss下降最大的处理
            #! 不到每类的最大个数，就直接把下降的loss和图片加进list中
            if len(self.class_decreased_most_loss[label]) < max_class_buffer_size:
                self.class_decreased_most_loss[label].append(decreased_loss[i].item())
                self.class_most_images[label].append(examples[i])
                self.class_most_adv_images[label].append(adv_aug_inputs[i])
            #! 达到每类最大个数后，则进行判断
            else:
                #! 若当前下降的损失大于list中的最大值，则将当前图片和损失替换进list
                if decreased_loss[i].item() > min(self.class_decreased_most_loss[label]):
                    idx = self.class_decreased_most_loss[label].index(min(self.class_decreased_most_loss[label]))
                    self.class_decreased_most_loss[label][idx] = decreased_loss[i].item()
                    self.class_most_images[label][idx] = examples[i]
                    self.class_most_adv_images[label][idx] = adv_aug_inputs[i]
    def add_low_high_decrease(self, n_tasks, add_adv:bool, percent:float, task, folder):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param percent: the percent of low loss, 1-percent of high loss
        :param task: current task number to save images
        :param folder: the folder to save images
        :return:
        """
        max_class_buffer_size = self.buffer_size // self.num_classes
        #! 更新buffer_count
        self.num_seen_examples += self.buffer_size // n_tasks
        low_num = int(math.ceil(max_class_buffer_size * percent))
        # high_num = max_class_buffer_size - low_num
        not_aug_inputs = []
        adv_aug_inputs = []
        min_label = min(self.current_task_labels)
        max_label = max(self.current_task_labels)
        for i in range(min_label,max_label+1):
            label_int = i
            label = str(label_int)

            for j in range(label_int * max_class_buffer_size,\
                        label_int * max_class_buffer_size + low_num):
                if add_adv:
                    self.examples[j] = self.class_adv_images[label][j % max_class_buffer_size]
                    not_aug_inputs.append(self.class_images[label][j % max_class_buffer_size])
                else:
                    self.examples[j] = self.class_images[label][j % max_class_buffer_size]
                    adv_aug_inputs.append(self.class_adv_images[label][j % max_class_buffer_size])
                self.labels[j] = torch.tensor(label_int).to(self.device)
            for j in range(label_int * max_class_buffer_size + low_num,\
                        (label_int+1) * max_class_buffer_size):
                if add_adv:
                    self.examples[j] = self.class_most_adv_images[label][j % max_class_buffer_size]
                    not_aug_inputs.append(self.class_most_images[label][j % max_class_buffer_size-low_num])
                else:
                    self.examples[j] = self.class_most_images[label][j % max_class_buffer_size-low_num]
                    adv_aug_inputs.append(self.class_most_adv_images[label][j % max_class_buffer_size-low_num])
                self.labels[j] = torch.tensor(label_int).to(self.device)
        
        #! 保存该task图片，可视化
        for i in range(min_label,max_label+1):
            for j in range(max_class_buffer_size):
                if add_adv:
                    adv_aug_inputs.append(self.examples[i*max_class_buffer_size+j])
                else:
                    not_aug_inputs.append(self.examples[i*max_class_buffer_size+j])
        not_aug_inputs = torch.stack(not_aug_inputs)
        adv_aug_inputs = torch.stack(adv_aug_inputs)
        # save_images(not_aug_inputs, adv_aug_inputs, task, folder)
        #! 一个task结束后清空缓存器
        self.current_task_labels = []



    def reservoir_class(self, label, max_class_buffer_size, abs_add):
        """
        Reservoir sampling algorithm.
        :return: the target index if the current image is sampled, else -1
        """      
        if self.num_seen_class_examples[label] < max_class_buffer_size:
            return self.num_seen_class_examples[label], 1
        if abs_add:
            rand = np.random.randint(0, self.num_seen_class_examples[label])
        else:
            rand = np.random.randint(0, self.num_seen_class_examples[label] + 1)
        if rand < max_class_buffer_size:
            return rand, 0
        else:
            return -1, 0

    def add_data_balanceclass(
        self, examples, labels=None, logits=None, task_labels=None, features=None, abs_add=False
    ):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
                :param features: tensor containing the latent features
        :return:
        """
        # if self.buffer_size == 20:
        #     time.sleep(100)
        if not hasattr(self, "examples"):
            self.init_tensors(examples, labels, logits, task_labels, features)

        max_class_buffer_size = self.buffer_size // self.num_classes
        for i in range(examples.shape[0]):
            # print(examples.shape[0])
            # index = reservoir(self.num_seen_examples, self.buffer_size)
            if labels is not None:
                index, count = self.reservoir_class(str(labels[i].item()), max_class_buffer_size, abs_add)
                self.num_seen_examples += count
            self.num_seen_class_examples[str(labels[i].item())] += 1
            if index >= 0:
                index = index + labels[i] * max_class_buffer_size
                index = int(index)
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)
                if features is not None:
                    self.features[index] = task_labels[i].to(self.device)
    



def save_images(not_aug_inputs, adv_aug_inputs, task, folder):
    images = not_aug_inputs[:not_aug_inputs.shape[0], :, :, :].cpu()
    watermarked_images = adv_aug_inputs[:adv_aug_inputs.shape[0], :, :, :].cpu()
    
    if adv_aug_inputs is not None:
    #! 对抗噪声可视化
        gap = images - watermarked_images
        gap = 15 * torch.abs(gap)
        stacked_images = torch.cat([images, watermarked_images,gap], dim=0)
    else:
        stacked_images = images 
    filename = os.path.join(folder, 'task-{}.png'.format(task))
    #! nrow (int, optional): Number of images displayed in each row of the grid.
    #! 此处即为每行original_images.shape[0]=batch数
    torchvision.utils.save_image(stacked_images, filename, nrow=not_aug_inputs.shape[0], normalize=False)