import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import numpy as np
from scipy import stats as st
import random
from torch.autograd import Variable
import os
import torchvision
# from hessian import hutchinson_trace_hvp
from mydatasets.utils.continual_dataset import ContinualDataset
from torch.autograd import grad
# from dct import *

def grad_loss(model, x, labels):
    cri = nn.CrossEntropyLoss(reduction='none')
    x = x.requires_grad_()
    logit = model(x)
    probs = torch.softmax(logit, dim=-1)
    loss = cri(logit, labels)
    grads = grad(loss, x, grad_outputs=torch.ones_like(loss), create_graph=True)[0]
    loss = loss + 0.1 * torch.norm(grads.view(x.shape[0], -1), dim=-1)
    return loss.mean()

def div_loss(model, x):
    """
    Compute the gradient of logsoftmax probabilities with respect to the input.

    Parameters:
    - model: PyTorch model
    - x: Input tensor
    - y: True labels

    Returns:
    - gradient: Gradient of logsoftmax probabilities with respect to input, flattened
    """

    model.eval()
    B, C, H, W = x.shape
    # x.requires_grad = True  # Enable gradient computation for input
    # print(x.shape)
    # # Forward pass
    logits = model(x)
    softmax_probs = torch.nn.functional.softmax(logits, dim=1)
    logsoftmax_probs = torch.log(softmax_probs)
    # Compute gradient of logsoftmax probabilities with respect to input
    # grad_logsoftmax = torch.zeros(x.shape[0], logits.shape[1], *x.shape[1:]).to(x.device)
    # grad_logsoftmax = grad(logsoftmax_probs, x, grad_outputs=torch.ones_like(logsoftmax_probs))[0]
    
    # Flatten the gradient
    # gradient = grad_logsoftmax.view(x.shape[0], -1)
    # print(softmax_probs.shape, grad_logsoftmax.shape)
    inf = softmax_probs * logsoftmax_probs
    inf = torch.sum(inf, dim = -1).mean()
    # print(inf.item())
    return inf

def save_images(not_aug_inputs, adv_aug_inputs, task, folder, resize_to=None):
    images = not_aug_inputs[:not_aug_inputs.shape[0], :, :, :].cpu()
    watermarked_images = adv_aug_inputs[:adv_aug_inputs.shape[0], :, :, :].cpu()

    #! 对抗噪声可视化
    gap = images - watermarked_images

    # # 对 width 和 height 维度进行最大-最小归一化
    # min_values, _ = torch.min(gap, dim=(2, 3), keepdim=True)
    # max_values, _ = torch.max(gap, dim=(2, 3), keepdim=True)
    # # 归一化操作
    # gap_normalized = (gap - min_values) / (max_values - min_values)
    gap = 15 * torch.abs(gap)

    if resize_to is not None:
        images = F.interpolate(images, size=resize_to)
        watermarked_images = F.interpolate(watermarked_images, size=resize_to)

    stacked_images = torch.cat([images, watermarked_images,gap], dim=0)
    filename = os.path.join(folder, 'task-{}.png'.format(task))
    #! nrow (int, optional): Number of images displayed in each row of the grid.
    #! 此处即为每行original_images.shape[0]=batch数
    torchvision.utils.save_image(stacked_images, filename, nrow=not_aug_inputs.shape[0], normalize=False)

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
    return outputs

class Attack(object):
    r"""
    Base class for all attacks.

    .. note::
        It automatically set device to the device where given model is.
        It basically changes training mode to eval during attack process.
        To change this, please see `set_training_mode`.
    """
    def __init__(self, name, model):
        r"""
        Initializes internal attack state.

        Arguments:
            name (str): name of attack.
            model (torch.nn.Module): model to attack.
        """

        self.attack = name
        self.model = model
        self.model_name = str(model).split("(")[0]
        self.device = next(model.parameters()).device

        self._attack_mode = 'default'
        self._targeted = False
        self._return_type = 'float'
        self._supported_mode = ['default']

        self._model_training = False
        self._batchnorm_training = False
        self._dropout_training = False

    def forward(self, *input):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def get_mode(self):
        r"""
        Get attack mode.

        """
        return self._attack_mode

    def set_mode_default(self):
        r"""
        Set attack mode as default mode.

        """
        self._attack_mode = 'default'
        self._targeted = False
        print("Attack mode is changed to 'default.'")

    def set_mode_targeted_by_function(self, target_map_function=None):
        r"""
        Set attack mode as targeted.

        Arguments:
            target_map_function (function): Label mapping function.
                e.g. lambda images, labels:(labels+1)%10.
                None for using input labels as targeted labels. (Default)

        """
        if "targeted" not in self._supported_mode:
            raise ValueError("Targeted mode is not supported.")

        self._attack_mode = 'targeted'
        self._targeted = True
        self._target_map_function = target_map_function
        print("Attack mode is changed to 'targeted.'")

    def set_mode_targeted_least_likely(self, kth_min=1):
        r"""
        Set attack mode as targeted with least likely labels.
        Arguments:
            kth_min (str): label with the k-th smallest probability used as target labels. (Default: 1)

        """
        if "targeted" not in self._supported_mode:
            raise ValueError("Targeted mode is not supported.")

        self._attack_mode = "targeted(least-likely)"
        self._targeted = True
        assert (kth_min > 0)
        self._kth_min = kth_min
        self._target_map_function = self._get_least_likely_label
        print("Attack mode is changed to 'targeted(least-likely).'")

    def set_mode_targeted_random(self):
        r"""
        Set attack mode as targeted with random labels.
        Arguments:
            num_classses (str): number of classes.

        """
        if "targeted" not in self._supported_mode:
            raise ValueError("Targeted mode is not supported.")

        self._attack_mode = "targeted(random)"
        self._targeted = True
        self._target_map_function = self._get_random_target_label
        print("Attack mode is changed to 'targeted(random).'")

    def set_return_type(self, type):
        r"""
        Set the return type of adversarial images: `int` or `float`.

        Arguments:
            type (str): 'float' or 'int'. (Default: 'float')

        .. note::
            If 'int' is used for the return type, the file size of 
            adversarial images can be reduced (about 1/4 for CIFAR10).
            However, if the attack originally outputs float adversarial images
            (e.g. using small step-size than 1/255), it might reduce the attack
            success rate of the attack.

        """
        if type == 'float':
            self._return_type = 'float'
        elif type == 'int':
            self._return_type = 'int'
        else:
            raise ValueError(type + " is not a valid type. [Options: float, int]")

    def set_training_mode(self, model_training=False, batchnorm_training=False, dropout_training=False):
        r"""
        Set training mode during attack process.

        Arguments:
            model_training (bool): True for using training mode for the entire model during attack process.
            batchnorm_training (bool): True for using training mode for batchnorms during attack process.
            dropout_training (bool): True for using training mode for dropouts during attack process.

        .. note::
            For RNN-based models, we cannot calculate gradients with eval mode.
            Thus, it should be changed to the training mode during the attack.
        """
        self._model_training = model_training
        self._batchnorm_training = batchnorm_training
        self._dropout_training = dropout_training

    def save(self, data_loader, save_path=None, verbose=True, return_verbose=False, save_pred=False):
        r"""
        Save adversarial images as torch.tensor from given torch.utils.data.DataLoader.

        Arguments:
            save_path (str): save_path.
            data_loader (torch.utils.data.DataLoader): data loader.
            verbose (bool): True for displaying detailed information. (Default: True)
            return_verbose (bool): True for returning detailed information. (Default: False)
            save_pred (bool): True for saving predicted labels (Default: False)

        """
        if save_path is not None:
            image_list = []
            label_list = []
            if save_pred:
                pre_list = []

        correct = 0
        total = 0
        l2_distance = []

        total_batch = len(data_loader)

        given_training = self.model.training
        given_return_type = self._return_type
        self._return_type = 'float'

        for step, (images, labels) in enumerate(data_loader):
            start = time.time()
            adv_images = self.__call__(images, labels)

            batch_size = len(images)

            if verbose or return_verbose:
                with torch.no_grad():
                    if given_training:
                        self.model.eval()
                    outputs = self.model(adv_images)
                    _, pred = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    right_idx = (pred == labels.to(self.device))
                    correct += right_idx.sum()
                    end = time.time()
                    delta = (adv_images - images.to(self.device)).view(batch_size, -1)
                    l2_distance.append(torch.norm(delta[~right_idx], p=2, dim=1))

                    rob_acc = 100 * float(correct) / total
                    l2 = torch.cat(l2_distance).mean().item()
                    progress = (step+1)/total_batch*100
                    elapsed_time = end-start
                    if verbose:
                        self._save_print(progress, rob_acc, l2, elapsed_time, end='\r')

            if save_path is not None:
                if given_return_type == 'int':
                    adv_images = self._to_uint(adv_images.detach().cpu())
                    image_list.append(adv_images)
                else:
                    image_list.append(adv_images.detach().cpu())
                label_list.append(labels.detach().cpu())

                image_list_cat = torch.cat(image_list, 0)
                label_list_cat = torch.cat(label_list, 0)

                if save_pred:
                    pre_list.append(pred.detach().cpu())
                    pre_list_cat = torch.cat(pre_list, 0)
                    torch.save((image_list_cat, label_list_cat, pre_list_cat), save_path)
                else:
                    torch.save((image_list_cat, label_list_cat), save_path)

        # To avoid erasing the printed information.
        if verbose:
            self._save_print(progress, rob_acc, l2, elapsed_time, end='\n')

        if given_training:
            self.model.train()

        if return_verbose:
            return rob_acc, l2, elapsed_time

    def _save_print(self, progress, rob_acc, l2, elapsed_time, end):
        print('- Save progress: %2.2f %% / Robust accuracy: %2.2f %% / L2: %1.5f (%2.3f it/s) \t' \
              % (progress, rob_acc, l2, elapsed_time), end=end)

    @torch.no_grad()
    def _get_target_label(self, images, labels=None):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        if self._targeted:
            given_training = self.model.training
            if given_training:
                self.model.eval()
            target_labels = self._target_map_function(images, labels)
            if given_training:
                self.model.train()
            return target_labels
        else:
            raise ValueError('Please define target_map_function.')

    @torch.no_grad()
    def _get_least_likely_label(self, images, labels=None):
        outputs = self.model(images)
        if labels is None:
            _, labels = torch.max(outputs, dim=1)
        n_classses = outputs.shape[-1]

        target_labels = torch.zeros_like(labels)
        for counter in range(labels.shape[0]):
            l = list(range(n_classses))
            l.remove(labels[counter])
            _, t = torch.kthvalue(outputs[counter][l], self._kth_min)
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device)

    @torch.no_grad()
    def _get_random_target_label(self, images, labels=None):
        outputs = self.model(images)
        if labels is None:
            _, labels = torch.max(outputs, dim=1)
        n_classses = outputs.shape[-1]

        target_labels = torch.zeros_like(labels)
        for counter in range(labels.shape[0]):
            l = list(range(n_classses))
            l.remove(labels[counter])
            t = (len(l)*torch.rand([1])).long().to(self.device)
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device)

    def _to_uint(self, images):
        r"""
        Function for changing the return type.
        Return images as int.
        """
        return (images*255).type(torch.uint8)

    def __str__(self):
        info = self.__dict__.copy()

        del_keys = ['model', 'attack']

        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys:
            del info[key]

        info['attack_mode'] = self._attack_mode
        info['return_type'] = self._return_type

        return self.attack + "(" + ', '.join('{}={}'.format(key, val) for key, val in info.items()) + ")"

    def __call__(self, *input, **kwargs):
        given_training = self.model.training

        if self._model_training:
            self.model.train()
            for _, m in self.model.named_modules():
                if not self._batchnorm_training:
                    if 'BatchNorm' in m.__class__.__name__:
                        m = m.eval()
                if not self._dropout_training:
                    if 'Dropout' in m.__class__.__name__:
                        m = m.eval()
        else:
            self.model.eval()

        images = self.forward(*input, **kwargs)

        if given_training:
            self.model.train()

        if self._return_type == 'int':
            images = self._to_uint(images)

        return images

class MIFGSM(Attack):
    r"""
    MI-FGSM in the paper 'Boosting Adversarial Attacks with Momentum'
    [https://arxiv.org/abs/1710.06081]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 1.0)
        steps (int): number of iterations. (Default: 5)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.MIFGSM(model, eps=8/255, steps=5, decay=1.0)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8/255, alpha=2/255, steps=5, decay=1.0,loss=nn.CrossEntropyLoss(),\
                 targeted=False, target=None, normalize = True, l2 = False):
        super().__init__("MIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
        self.l2 = l2
    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # if self.targeted:
        #     labels = self.target*torch.ones_like(labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = self.loss

        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()

        for _ in range(self.steps):
            #adv_images.requires_grad = True
            adv_delta.requires_grad = True
            adv_images = torch.clamp(clean_images+adv_delta, min=0, max=1)
            if self.normalize:
                normalized_adv_images = self.norm(adv_images)
                outputs = self.model(normalized_adv_images)
            else:
                outputs = self.model(adv_images)
            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_delta,
                                       retain_graph=False, create_graph=False)[0]
            if self.l2:
                grad = grad/grad.norm(p=2)
                grad = grad + momentum*self.decay
                momentum = grad

                adv_delta = adv_delta.detach() + self.alpha * grad.sign()
                current_l2_norm = adv_delta.norm(p=2)
                if current_l2_norm > self.eps:
                    # 将扰动向量除以其l2范数后乘以最大允许的l2范数
                    adv_delta = adv_delta / current_l2_norm * self.eps
            else:
                grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
                grad = grad + momentum*self.decay
                momentum = grad

                adv_delta = adv_delta.detach() + self.alpha*grad.sign()
                adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images

class TIFGSM(Attack):
    r"""
    TIFGSM in the paper 'Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks'
    [https://arxiv.org/abs/1904.02884]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of iterations. (Default: 20)
        decay (float): momentum factor. (Default: 0.0)
        kernel_name (str): kernel name. (Default: gaussian)
        len_kernel (int): kernel length.  (Default: 15, which is the best according to the paper)
        nsig (int): radius of gaussian kernel. (Default: 3; see Section 3.2.2 in the paper for explanation)
        resize_rate (float): resize factor used in input diversity. (Default: 0.9)
        diversity_prob (float) : the probability of applying input diversity. (Default: 0.5)
        random_start (bool): using random initialization of delta. (Default: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.TIFGSM(model, eps=8/255, alpha=2/255, steps=20, decay=1.0, resize_rate=0.9, diversity_prob=0.7, random_start=False)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=8/255, alpha=2/255, steps=20, decay=1.0, kernel_name='gaussian',
                 len_kernel=5, nsig=3, resize_rate=0.9, diversity_prob=0.7, random_start=False,
                 loss=nn.CrossEntropyLoss(),targeted=False, target=None,normalize=True):
        super().__init__("TIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        loss = self.loss
        momentum = torch.zeros_like(images).detach().to(self.device)
        stacked_kernel = self.stacked_kernel.to(self.device)

        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_delta = adv_delta + torch.empty_like(adv_delta).uniform_(-self.eps, self.eps)
            clean_images = torch.clamp(clean_images+adv_delta, min=0, max=1).detach()

        for _ in range(self.steps):
            #adv_images.requires_grad = True
            adv_delta.requires_grad = True
            adv_images = torch.clamp(clean_images+adv_delta, min=0, max=1)
            if self.normalize:
                outputs = self.model(self.norm(self.input_diversity(adv_images)))
            else:
                outputs = self.model(self.input_diversity(adv_images))

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_delta,
                                       retain_graph=False, create_graph=False)[0]
            # depth wise conv2d
            grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad
            # adv_delta = adv_delta.detach() + self.alpha * grad/grad.norm(p=2)#.sign()
            
            # current_l2_norm = adv_delta.norm(p=2)
            # if current_l2_norm > self.eps:
            #     # 将扰动向量除以其l2范数后乘以最大允许的l2范数
            #     adv_delta = adv_delta / current_l2_norm * self.eps

            adv_delta = adv_delta.detach() + self.alpha*grad.sign()
            adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x

import math
class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / \
            (running_mean + 1e-6) / input.shape[0]
        return grad, None


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)

    # Recalculate ema

    return t_log, running_mean


class Mine(nn.Module):
    def __init__(self, loss='mine', alpha=0.01):
        super().__init__()
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha

        self.T = nn.Sequential(nn.Conv2d(6, 32, 5, 2, 2), nn.LeakyReLU(), 
                               nn.Conv2d(32, 64, 5, 2, 2), nn.LeakyReLU(), 
                               nn.Conv2d(64, 128, 5, 2 ,2), nn.LeakyReLU(),
                               nn.Conv2d(128, 256, 5, 2, 2), nn.LeakyReLU(),
                               nn.Conv2d(256, 512, 5, 2, 2), nn.LeakyReLU(),
                               nn.Conv2d(512, 1024, 5, 2, 2), nn.LeakyReLU(),
                               nn.AdaptiveAvgPool2d(1)
                               )
        self.Linear = nn.Sequential(nn.Linear(1024, 1),
                               nn.Sigmoid())

    def forward(self, x, z, z_marg=None):
        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])]

        t = self.Linear(self.T(torch.cat((x, z), dim = 1)).view(-1,1)).mean()
        t_marg = self.Linear(self.T(torch.cat((x, z_marg) ,dim = 1)).view(-1,1))

        if self.loss in ['mine']:
            second_term, self.running_mean = ema_loss(
                t_marg, self.running_mean, self.alpha)
        elif self.loss in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()
        elif self.loss in ['mine_biased']:
            second_term = torch.logsumexp(
                t_marg, 0) - math.log(t_marg.shape[0])

        return -t + second_term

    def mi(self, x, z, z_marg=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()

        with torch.no_grad():
            mi = -self.forward(x, z, z_marg)
        return mi

class IB_TIFGSM(Attack):
   
    def __init__(self, model, eps=8/255, alpha=2/255, steps=20, decay=1.0, kernel_name='gaussian',
                 len_kernel=5, nsig=3, resize_rate=0.9, diversity_prob=0.7, random_start=False,
                 loss=nn.CrossEntropyLoss(),targeted=False, target=None,normalize=True):
        super().__init__("IB_TIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
        self.mine = self.load_MINE().to(self.device)

    def load_MINE(self):
        mine = torch.load('MINE')
        #mine.load_state_dict()
        return mine
    
    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # if self.targeted:
        #     labels = self.target*torch.ones_like(labels)

        loss = self.loss
        momentum = torch.zeros_like(images).detach().to(self.device)
        stacked_kernel = self.stacked_kernel.to(self.device)

        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_delta = adv_delta + torch.empty_like(adv_delta).uniform_(-self.eps, self.eps)
            clean_images = torch.clamp(clean_images+adv_delta, min=0, max=1).detach()

        for _ in range(self.steps):
            #adv_images.requires_grad = True
            adv_delta.requires_grad = True
            adv_images = torch.clamp(clean_images+adv_delta, min=0, max=1)
            if self.normalize:
                outputs = self.model(self.norm(self.input_diversity(adv_images)))
            else:
                outputs = self.model(self.input_diversity(adv_images))

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, labels) - self.mine(clean_images, adv_delta)
            else:
                cost = loss(outputs, labels) + self.mine(clean_images, adv_delta)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_delta,
                                       retain_graph=False, create_graph=False)[0]
            # depth wise conv2d
            grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_delta = adv_delta.detach() + self.alpha*grad.sign()
            adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x

class Logits_TI(Attack):

    def __init__(self, model, eps=8/255, alpha=2/255, steps=20, decay=1.0, kernel_name='gaussian',
                 len_kernel=5, nsig=3, resize_rate=0.9, diversity_prob=0.7, random_start=False,
                 loss=nn.CrossEntropyLoss(),targeted=False, target=None,normalize=True):
        super().__init__("Logits_TI", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize

    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t
    
    def logit_loss(self, out, labels):
        return -torch.gather(out, 1, labels.view(-1, 1)).squeeze(-1).mean()

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # if self.targeted:
        #     labels = self.target*torch.ones_like(labels)

        #loss = self.loss
        momentum = torch.zeros_like(images).detach().to(self.device)
        stacked_kernel = self.stacked_kernel.to(self.device)

        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_delta = adv_delta + torch.empty_like(adv_delta).uniform_(-self.eps, self.eps)
            clean_images = torch.clamp(clean_images+adv_delta, min=0, max=1).detach()

        for _ in range(self.steps):
            #adv_images.requires_grad = True
            adv_delta.requires_grad = True
            adv_images = torch.clamp(clean_images+adv_delta, min=0, max=1)
            if self.normalize:
                outputs = self.model(self.norm(self.input_diversity(adv_images)))
            else:
                outputs = self.model(self.input_diversity(adv_images))

            # Calculate loss
            if self.targeted:
                cost = -self.logit_loss(outputs, labels)
            else:
                cost = self.logit_loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_delta,
                                       retain_graph=False, create_graph=False)[0]
            # depth wise conv2d
            grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_delta = adv_delta.detach() + self.alpha*grad.sign()
            adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x

class Bo_Logits_TI(Attack):

    def __init__(self, model, eps=8/255, alpha=2/255, steps=20, decay=1.0, kernel_name='gaussian',
                 len_kernel=5, nsig=3, resize_rate=0.9, diversity_prob=0.7, random_start=False,
                 loss=nn.CrossEntropyLoss(),targeted=False, target=None,normalize=True, gamma=0.1, lmd=0.01, sigma=0.1):
        super().__init__("Bo_Logits_TI", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
        self.gamma = gamma
        self.lmd = lmd
        self.sigma = sigma

    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t
    
    def logit_loss(self, out, labels):
        return -torch.gather(out, 1, labels.view(-1, 1)).squeeze(-1).mean()

    def double_kl_div(self, p_output, q_output, get_softmax=True,gamma=0.01):

        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        
        return  gamma*KLDivLoss(F.log_softmax(p_output,dim=-1), F.softmax(q_output,dim=-1))+KLDivLoss(F.log_softmax(q_output,dim=-1), F.softmax(p_output,dim=-1))

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # if self.targeted:
        #     labels = self.target*torch.ones_like(labels)

        #loss = self.loss
        momentum = torch.zeros_like(images).detach().to(self.device)
        stacked_kernel = self.stacked_kernel.to(self.device)

        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()
        clean_images_shifted = clean_images+self.sigma*(torch.randn_like(clean_images))
        clean_images_shifted = torch.clamp(clean_images_shifted, min=0, max=1)

        if self.random_start:
            # Starting at a uniformly random point
            adv_delta = adv_delta + torch.empty_like(adv_delta).uniform_(-self.eps, self.eps)
            clean_images = torch.clamp(clean_images+adv_delta, min=0, max=1).detach()

        for _ in range(self.steps):
            #adv_images.requires_grad = True
            adv_delta.requires_grad = True
            adv_images = torch.clamp(clean_images+adv_delta, min=0, max=1)
            adv_images_shifted = torch.clamp(clean_images_shifted+adv_delta, min=0, max=1)

            if self.normalize:
                outputs = self.model(self.norm(self.input_diversity(adv_images)))
                outputs_shifted = self.model(self.norm(self.input_diversity(adv_images_shifted)))
            else:
                outputs = self.model(self.input_diversity(adv_images))
                outputs_shifted = self.model(self.input_diversity(adv_images_shifted))

            # Calculate loss
            if self.targeted:
                cost = -self.logit_loss(outputs, labels)-self.gamma*self.logit_loss(outputs_shifted, labels)-self.lmd*self.double_kl_div(outputs, outputs_shifted,gamma=self.gamma)
            else:
                cost = self.logit_loss(outputs, labels)+self.gamma*self.logit_loss(outputs_shifted, labels)-self.lmd*self.double_kl_div(outputs, outputs_shifted,gamma=self.gamma)
            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_delta,
                                       retain_graph=False, create_graph=False)[0]
            # depth wise conv2d
            grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_delta = adv_delta.detach() + self.alpha*grad.sign()
            adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x

class RAP_Logits_TI(Attack):

    def __init__(self, model, eps=8/255, alpha=2/255, steps=20, decay=1.0, kernel_name='gaussian',
                 len_kernel=5, nsig=3, resize_rate=0.9, diversity_prob=0.7, random_start=False,
                 loss=nn.CrossEntropyLoss(),targeted=False, target=None,normalize=True, k_ls = 100, epsilon_n = 12):
        super().__init__("RAP_Logits_TI", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
        self.k_ls = k_ls
        self.epsilon_n = epsilon_n

    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t
    
    def logit_loss(self, out, labels):
        return -torch.gather(out, 1, labels.view(-1, 1)).squeeze(-1).mean()

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # if self.targeted:
        #     labels = self.target*torch.ones_like(labels)

        #loss = self.loss
        momentum = torch.zeros_like(images).detach().to(self.device)
        stacked_kernel = self.stacked_kernel.to(self.device)

        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_delta = adv_delta + torch.empty_like(adv_delta).uniform_(-self.eps, self.eps)
            clean_images = torch.clamp(clean_images+adv_delta, min=0, max=1).detach()

        

        for _ in range(self.steps):
            #adv_images.requires_grad = True
            
            
            if _ >= self.k_ls:
                
                n_rap = torch.zeros_like(images).to(self.device)
                
                momentum_rap = torch.zeros_like(images).detach().to(self.device)
                for i in range(1):
                    n_rap.requires_grad = True
                    adv_images = torch.clamp(clean_images+adv_delta+n_rap, min=0, max=1)
                    if self.normalize:
                        outputs = self.model(self.norm(self.input_diversity(adv_images)))
                    else:
                        outputs = self.model(self.input_diversity(adv_images))
                    # Calculate loss
                    if self.targeted:
                        cost = self.logit_loss(outputs, labels)
                    else:
                        cost = -self.logit_loss(outputs, labels)
                    grad = torch.autograd.grad(cost, n_rap,
                                            retain_graph=False, create_graph=False)[0]
                    # depth wise conv2d
                    grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)
                    grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
                    grad = grad + momentum_rap*self.decay
                    momentum_rap = grad

                    n_rap = n_rap.detach() + self.alpha*grad.sign()
                    n_rap = torch.clamp(n_rap, min=-self.epsilon_n, max=self.epsilon_n)
            
            for j in range(1):
                adv_delta.requires_grad = True
                if _ >= self.k_ls:
                    adv_images = torch.clamp(clean_images+adv_delta+n_rap, min=0, max=1)
                else:
                    adv_images = torch.clamp(clean_images+adv_delta, min=0, max=1)
                if self.normalize:
                    outputs = self.model(self.norm(self.input_diversity(adv_images)))
                else:
                    outputs = self.model(self.input_diversity(adv_images))

                # Calculate loss
                if self.targeted:
                    cost = -self.logit_loss(outputs, labels)
                else:
                    cost = self.logit_loss(outputs, labels)

                # Update adversarial images
                grad = torch.autograd.grad(cost, adv_delta,
                                        retain_graph=False, create_graph=False)[0]
                # depth wise conv2d
                grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)
                grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
                grad = grad + momentum*self.decay
                momentum = grad

                adv_delta = adv_delta.detach() + self.alpha*grad.sign()
                adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x

class Bo_RAP_Logits_TI(Attack):

    def __init__(self, model, eps=8/255, alpha=2/255, steps=20, decay=1.0, kernel_name='gaussian',
                 len_kernel=5, nsig=3, resize_rate=0.9, diversity_prob=0.7, random_start=False,
                 loss=nn.CrossEntropyLoss(),targeted=False, target=None,normalize=True, k_ls = 100, epsilon_n = 12, gamma=0.1, lmd=0.01, sigma=0.1):
        super().__init__("Bo_RAP_Logits_TI", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
        self.k_ls = k_ls
        self.epsilon_n = epsilon_n
        self.gamma = gamma
        self.lmd = lmd
        self.sigma = sigma

    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t
    
    def logit_loss(self, out, labels):
        return -torch.gather(out, 1, labels.view(-1, 1)).squeeze(-1).mean()

    def double_kl_div(self, p_output, q_output, get_softmax=True,gamma=0.01):

        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        
        return  gamma*KLDivLoss(F.log_softmax(p_output,dim=-1), F.softmax(q_output,dim=-1))+KLDivLoss(F.log_softmax(q_output,dim=-1), F.softmax(p_output,dim=-1))

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # if self.targeted:
        #     labels = self.target*torch.ones_like(labels)

        #loss = self.loss
        momentum = torch.zeros_like(images).detach().to(self.device)
        stacked_kernel = self.stacked_kernel.to(self.device)

        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()
        clean_images_shifted = clean_images+self.sigma*(torch.randn_like(clean_images))
        clean_images_shifted = torch.clamp(clean_images_shifted, min=0, max=1)
        if self.random_start:
            # Starting at a uniformly random point
            adv_delta = adv_delta + torch.empty_like(adv_delta).uniform_(-self.eps, self.eps)
            clean_images = torch.clamp(clean_images+adv_delta, min=0, max=1).detach()

        for _ in range(self.steps):
            #adv_images.requires_grad = True
            
            
            if _ >= self.k_ls:
                
                n_rap = torch.zeros_like(images).to(self.device)
                
                momentum_rap = torch.zeros_like(images).detach().to(self.device)
                for i in range(1):
                    n_rap.requires_grad = True
                    adv_images = torch.clamp(clean_images+adv_delta+n_rap, min=0, max=1)
                    if self.normalize:
                        outputs = self.model(self.norm(self.input_diversity(adv_images)))
                    else:
                        outputs = self.model(self.input_diversity(adv_images))
                    # Calculate loss
                    if self.targeted:
                        cost = self.logit_loss(outputs, labels)
                    else:
                        cost = -self.logit_loss(outputs, labels)
                    grad = torch.autograd.grad(cost, n_rap,
                                            retain_graph=False, create_graph=False)[0]
                    # depth wise conv2d
                    grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)
                    grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
                    grad = grad + momentum_rap*self.decay
                    momentum_rap = grad

                    n_rap = n_rap.detach() + self.alpha*grad.sign()
                    n_rap = torch.clamp(n_rap, min=-self.epsilon_n, max=self.epsilon_n)
            
            for j in range(1):
                adv_delta.requires_grad = True
                if _ >= self.k_ls:
                    adv_images = torch.clamp(clean_images+adv_delta+n_rap, min=0, max=1)
                    adv_images_shifted = torch.clamp(clean_images_shifted+adv_delta+n_rap, min=0, max=1)
                else:
                    adv_images = torch.clamp(clean_images+adv_delta, min=0, max=1)
                    adv_images_shifted = torch.clamp(clean_images_shifted+adv_delta, min=0, max=1)

                if self.normalize:
                    outputs = self.model(self.norm(self.input_diversity(adv_images)))
                    outputs_shifted = self.model(self.norm(self.input_diversity(adv_images_shifted)))
                else:
                    outputs = self.model(self.input_diversity(adv_images))
                    outputs_shifted = self.model(self.input_diversity(adv_images_shifted))
                # Calculate loss
                if self.targeted:
                    cost = -self.logit_loss(outputs, labels)-self.gamma*self.logit_loss(outputs_shifted, labels)-self.lmd*self.double_kl_div(outputs, outputs_shifted,gamma=self.gamma)
                else:
                    cost = self.logit_loss(outputs, labels)+self.gamma*self.logit_loss(outputs_shifted, labels)-self.lmd*self.double_kl_div(outputs, outputs_shifted,gamma=self.gamma)

                # Update adversarial images
                grad = torch.autograd.grad(cost, adv_delta,
                                        retain_graph=False, create_graph=False)[0]
                # depth wise conv2d
                grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)
                grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
                grad = grad + momentum*self.decay
                momentum = grad

                adv_delta = adv_delta.detach() + self.alpha*grad.sign()
                adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x

class S2I_MIFGSM(Attack):

    def __init__(self, model, eps=8/255, alpha=2/255, steps=5, decay=1.0,loss=nn.CrossEntropyLoss(),targeted=False, target=None, normalize = True, rho = 0.5, N = 20, sigma = 16/255, diversity_prob = 0.7, resize_rate = 0.9):
        super().__init__("S2I_MIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
        self.rho = rho
        self.N = N
        self.sigma = sigma
        self.diversity_prob = diversity_prob
        self.resize_rate = resize_rate

    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t

    def input_diversity(self, x):
            img_size = x.shape[-1]
            img_resize = int(img_size * self.resize_rate)

            if self.resize_rate < 1:
                img_size = img_resize
                img_resize = x.shape[-1]

            rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
            rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
            h_rem = img_resize - rnd
            w_rem = img_resize - rnd
            pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
            pad_bottom = h_rem - pad_top
            pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
            pad_right = w_rem - pad_left

            padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

            return padded if torch.rand(1) < self.diversity_prob else x    

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            labels = self.target*torch.ones_like(labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = self.loss

        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()

        for _ in range(self.steps):
            adv_delta.requires_grad = True
            adv_images = torch.clamp(clean_images+adv_delta, min=0, max=1)
            noise = 0
            for n in range(self.N):
                gauss = torch.randn_like(clean_images) * (self.sigma / 255)
                
                x_dct = dct_2d(clean_images + gauss)
                mask = (torch.rand_like(clean_images) * 2 * self.rho + 1 - self.rho).cuda()
                x_idct = idct_2d(x_dct * mask) + adv_delta
                
                # DI-FGSM https://arxiv.org/abs/1803.06978
                # output_v3 = model(DI(x_idct))

                if self.normalize:
                    normalized_adv_x = self.norm(self.input_diversity(x_idct))
                    outputs = self.model(normalized_adv_x)
                else:
                    outputs = self.model(self.input_diversity(x_idct))
                # Calculate loss
                if self.targeted:
                    cost = -loss(outputs, labels)
                else:
                    cost = loss(outputs, labels)

                # Update adversarial images
                grad_x = torch.autograd.grad(cost, adv_delta,
                                        retain_graph=False, create_graph=False)[0]
                grad_x = grad_x / torch.mean(torch.abs(grad_x), dim=(1,2,3), keepdim=True)
                
                noise += grad_x

            grad = noise / self.N
            grad = grad + momentum*self.decay
            momentum = grad

            adv_delta = adv_delta.detach() + self.alpha*grad.sign()
            adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        
            #adv_images.requires_grad = True
            

        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images

class Bo_S2I_MIFGSM(Attack):

    def __init__(self, model, eps=8/255, alpha=2/255, steps=5, decay=1.0,loss=nn.CrossEntropyLoss(),targeted=False, target=None, normalize = True, rho = 0.5, N = 20, sigma = 16/255, diversity_prob = 0.7, resize_rate = 0.9, gamma = 0.1, lmd=0.01, sig=0.1):
        super().__init__("Bo_S2I_MIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
        self.rho = rho
        self.N = N
        self.sigma = sigma
        self.diversity_prob = diversity_prob
        self.resize_rate = resize_rate
        self.gamma = gamma
        self.lmd = lmd
        self.sig = sig


    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t

    def double_kl_div(self, p_output, q_output, get_softmax=True,gamma=0.01):

        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        
        return  gamma*KLDivLoss(F.log_softmax(p_output,dim=-1), F.softmax(q_output,dim=-1))+KLDivLoss(F.log_softmax(q_output,dim=-1), F.softmax(p_output,dim=-1))

    def input_diversity(self, x):
            img_size = x.shape[-1]
            img_resize = int(img_size * self.resize_rate)

            if self.resize_rate < 1:
                img_size = img_resize
                img_resize = x.shape[-1]

            rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
            rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
            h_rem = img_resize - rnd
            w_rem = img_resize - rnd
            pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
            pad_bottom = h_rem - pad_top
            pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
            pad_right = w_rem - pad_left

            padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

            return padded if torch.rand(1) < self.diversity_prob else x    

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            labels = self.target*torch.ones_like(labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = self.loss

        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()
        clean_images_shifted = clean_images+self.sig*(torch.randn_like(clean_images))
        clean_images_shifted = torch.clamp(clean_images_shifted, min=0, max=1)

        for _ in range(self.steps):
            adv_delta.requires_grad = True
            adv_images = torch.clamp(clean_images+adv_delta, min=0, max=1)
            adv_images_shifted = torch.clamp(clean_images_shifted+adv_delta, min=0, max=1)
            noise = 0
            for n in range(self.N):
                gauss = torch.randn_like(clean_images) * (self.sigma / 255)
                
                x_dct = dct_2d(clean_images + gauss)
                x_dct_shifted = dct_2d(clean_images_shifted + gauss)
                mask = (torch.rand_like(clean_images) * 2 * self.rho + 1 - self.rho).cuda()
                x_idct = idct_2d(x_dct * mask) + adv_delta
                x_idct_shifted = idct_2d(x_dct_shifted * mask) + adv_delta

                # DI-FGSM https://arxiv.org/abs/1803.06978
                # output_v3 = model(DI(x_idct))
                if self.normalize:
                    outputs = self.model(self.norm(self.input_diversity(x_idct)))
                    outputs_shifted = self.model(self.norm(self.input_diversity(x_idct_shifted)))
                else:
                    outputs = self.model(self.input_diversity(x_idct))
                    outputs_shifted = self.model(self.input_diversity(x_idct_shifted))

                # Calculate loss
                if self.targeted:
                    cost = -loss(outputs, labels)-self.gamma*loss(outputs_shifted, labels)-self.lmd*self.double_kl_div(outputs, outputs_shifted,gamma=self.gamma)
                else:
                    cost = loss(outputs, labels)+self.gamma*loss(outputs_shifted, labels)-self.lmd*self.double_kl_div(outputs, outputs_shifted,gamma=self.gamma)

                # Update adversarial images
                grad_x = torch.autograd.grad(cost, adv_delta,
                                        retain_graph=False, create_graph=False)[0]
                grad_x = grad_x / torch.mean(torch.abs(grad_x), dim=(1,2,3), keepdim=True)
                
                noise += grad_x

            grad = noise / self.N
            grad = grad + momentum*self.decay
            momentum = grad

            adv_delta = adv_delta.detach() + self.alpha*grad.sign()
            adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        
            #adv_images.requires_grad = True
            

        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images

class Bo_DIFGSM(Attack):

    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, decay=1.0,
                 resize_rate=0.9, diversity_prob=0.7, random_start=False,loss=nn.CrossEntropyLoss(),targeted=False, target=None, normalize=True):
        super().__init__("Bo_DIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t
    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x

    def double_kl_div(self, p_output, q_output, get_softmax=True, gamma=0.01):

        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        
        return gamma*KLDivLoss(F.log_softmax(p_output,dim=-1), F.softmax(q_output,dim=-1))+KLDivLoss(F.log_softmax(q_output,dim=-1), F.softmax(p_output,dim=-1))

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            labels = self.target*torch.ones_like(labels)

        momentum = torch.zeros_like(images).detach().to(self.device)
        loss = self.loss
        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()
        clean_images_shifted = clean_images+0.1*(torch.randn_like(clean_images))
        clean_images_shifted = torch.clamp(clean_images_shifted, min=0, max=1).detach()
        if self.random_start:
            # Starting at a uniformly random point
            adv_delta = adv_delta + torch.empty_like(adv_delta).uniform_(-self.eps, self.eps)
            clean_images = torch.clamp(clean_images+adv_delta, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_delta.requires_grad = True
            adv_images = torch.clamp(clean_images+adv_delta, min=0, max=1)
            adv_images_shifted = torch.clamp(clean_images_shifted+adv_delta, min=0, max=1)
            
            if self.normalize:
                outputs = self.model(self.norm(self.input_diversity(adv_images)))
                outputs_shifted = self.model(self.norm(self.input_diversity(adv_images_shifted)))
            else:
                outputs = self.model(self.input_diversity(adv_images))
                outputs_shifted = self.model(adv_images_shifted)
            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, labels)-0.1*loss(outputs_shifted, labels)-0.01*self.double_kl_div(outputs, outputs_shifted, gamma=0.1)
            else:
                cost = loss(outputs, labels)+0.1*loss(outputs_shifted, labels)-0.01*self.double_kl_div(outputs, outputs_shifted, gamma=0.1)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_delta,
                                       retain_graph=False, create_graph=False)[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_delta = adv_delta.detach() + self.alpha*grad.sign()
            adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images

class Bo_MIFGSM(Attack):

    def __init__(self, model, eps=8/255, alpha=2/255, steps=5, decay=1.0,loss=nn.CrossEntropyLoss(),targeted=False, target=None, normalize = True, gamma=0.1, lmd=0.01, sigma=0.1):
        super().__init__("Bo_MIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
        self.gamma = gamma
        self.lmd = lmd
        self.sigma = sigma
    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t
    def double_kl_div(self, p_output, q_output, get_softmax=True,gamma=0.01):

        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        
        return  gamma*KLDivLoss(F.log_softmax(p_output,dim=-1), F.softmax(q_output,dim=-1))+KLDivLoss(F.log_softmax(q_output,dim=-1), F.softmax(p_output,dim=-1))
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            labels = self.target*torch.ones_like(labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = self.loss

        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()
        clean_images_shifted = clean_images+self.sigma*(torch.randn_like(clean_images))
        clean_images_shifted = torch.clamp(clean_images_shifted, min=0, max=1)
        for _ in range(self.steps):
            #adv_images.requires_grad = True
            adv_delta.requires_grad = True
            adv_images = torch.clamp(clean_images+adv_delta, min=0, max=1)
            adv_images_shifted = torch.clamp(clean_images_shifted+adv_delta, min=0, max=1)
            if self.normalize:
                outputs = self.model(self.norm(adv_images))
                outputs_shifted = self.model(self.norm(adv_images_shifted))
            else:
                outputs = self.model(adv_images)
                outputs_shifted = self.model(adv_images_shifted)
            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, labels)-self.gamma*loss(outputs_shifted, labels)-self.lmd*self.double_kl_div(outputs, outputs_shifted,gamma=self.gamma)
            else:
                cost = loss(outputs, labels)+self.gamma*loss(outputs_shifted, labels)-self.lmd*self.double_kl_div(outputs, outputs_shifted,gamma=self.gamma)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_delta,
                                       retain_graph=False, create_graph=False)[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_delta = adv_delta.detach() + self.alpha*grad.sign()
            adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images

class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=16/255,
                 alpha=2/255, steps=40, random_start=False,loss=nn.CrossEntropyLoss(),\
                 targeted=False, target=None,normalize = True, l2=False):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
        self.l2 = l2
    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # if self.targeted:
        #     labels = self.target*torch.ones_like(labels)

        loss = self.loss

        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_delta = adv_delta + torch.empty_like(adv_delta).uniform_(-self.eps, self.eps)
            # clean_images = torch.clamp(clean_images+adv_delta, min=0, max=1).detach()
        decay_factor = 0.8
        for _ in range(self.steps):

            adv_delta.requires_grad = True
            adv_images = torch.clamp(clean_images+adv_delta, min=0, max=1)
            if self.normalize:
                outputs = self.model(self.norm(adv_images))
            else:
                outputs = self.model(adv_images)
            # outputs = mask_classes(outputs, dataset, k)
            # Calculate loss
        
            if self.targeted:
                # cost = -grad_loss(self.model, adv_images, labels)
                # if _%6 == 0:
                #     cost = loss(outputs, labels)
                # else:
                    cost = -loss(outputs, labels)
            else:
                # if _%6 == 0:
                #     cost = -loss(outputs, labels)
                # else:
                    cost = loss(outputs, labels)
                # cost = grad_loss(self.model, adv_images, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_delta,
                                       retain_graph=False, create_graph=False)[0]
            if self.l2:
                adv_delta = adv_delta.detach() + self.alpha * grad/grad.norm(p=2)#.sign()
                
                current_l2_norm = adv_delta.norm(p=2)
                if current_l2_norm > self.eps:
                    # 将扰动向量除以其l2范数后乘以最大允许的l2范数
                    adv_delta = adv_delta / current_l2_norm * self.eps
                # adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
                # self.alpha *= decay_factor
            else:
                adv_delta = adv_delta.detach() + self.alpha*grad.sign()
                adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images

class NIFGSM(Attack):
    r"""
    NI-FGSM in the paper 'NESTEROV ACCELERATED GRADIENT AND SCALEINVARIANCE FOR ADVERSARIAL ATTACKS'
    [https://arxiv.org/abs/1908.06281], Published as a conference paper at ICLR 2020
    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 1.0)
        steps (int): number of iterations. (Default: 10)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = torchattacks.NIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=1.0)
        >>> adv_images = attack(images, labels)
    """

    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, decay=1.0,loss=nn.CrossEntropyLoss(),\
                    targeted=False, target=None, normalize = True,l2 = False):
        super().__init__("NIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        #! self.alpha = eps / steps
        self.supported_mode = ['default', 'targeted']
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
        self.l2 = l2
    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # if self.targeted:
        #     labels = self.target*torch.ones_like(labels)

        momentum = torch.zeros_like(images).detach().to(self.device)
        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()

        for _ in range(self.steps):
            #adv_images.requires_grad = True
            adv_delta.requires_grad = True
            adv_images = clean_images+adv_delta
            nes_images = torch.clamp(adv_images + self.decay*self.alpha*momentum, min=0, max=1)
            if self.normalize:
                outputs = self.model(self.norm(nes_images))
            else:
                outputs = self.model(nes_images)
            # Calculate loss
            if self.targeted:
                cost = -self.loss(outputs, labels)
            else:
                cost = self.loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_delta,
                                       retain_graph=False, create_graph=False)[0]
            if self.l2:
                grad = self.decay*momentum + grad/grad.norm(p=2)
                momentum = grad
                adv_delta = adv_delta.detach() + self.alpha * grad.sign()
                current_l2_norm = adv_delta.norm(p=2)
                if current_l2_norm > self.eps:
                    # 将扰动向量除以其l2范数后乘以最大允许的l2范数
                    adv_delta = adv_delta / current_l2_norm * self.eps
            else:
                grad = self.decay*momentum + grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
                momentum = grad
                adv_delta = adv_delta.detach() + self.alpha*grad.sign()
                adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images

class SINIFGSM(Attack):
    r"""
    SI-NI-FGSM in the paper 'NESTEROV ACCELERATED GRADIENT AND SCALEINVARIANCE FOR ADVERSARIAL ATTACKS'
    [https://arxiv.org/abs/1908.06281], Published as a conference paper at ICLR 2020
    Modified from "https://githuba.com/JHL-HUST/SI-NI-FGSM"
    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of iterations. (Default: 10)
        decay (float): momentum factor. (Default: 1.0)
        m (int): number of scale copies. (Default: 5)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = torchattacks.SINIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=1.0, m=5)
        >>> adv_images = attack(images, labels)
    """

    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, decay=1.0, m=5,loss=nn.CrossEntropyLoss(),targeted=False, target=None,normalize = True):
        super().__init__("SINIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.m = m
        self.supported_mode = ['default', 'targeted']
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # if self.targeted:
        #     labels = self.target*torch.ones_like(labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()

        for _ in range(self.steps):
            adv_delta.requires_grad = True
            adv_images = clean_images+adv_delta
            nes_images = torch.clamp(adv_images + self.decay*self.alpha*momentum, min=0, max=1)
    
            # Calculate sum the gradients over the scale copies of the input image
            adv_grad = torch.zeros_like(images).detach().to(self.device)
            for i in torch.arange(self.m):
                nes_images = nes_images / torch.pow(2, i)
                if self.normalize:
                    outputs = self.model(self.norm(nes_images))
                else:
                    outputs = self.model(nes_images)
                # Calculate loss
                if self.targeted:
                    cost = -self.loss(outputs, labels)
                else:
                    cost = self.loss(outputs, labels)
                adv_grad += torch.autograd.grad(cost, adv_delta,
                                                retain_graph=True, create_graph=True)[0].detach()
            adv_grad = adv_grad / self.m

            # Update adversarial images
            grad = self.decay*momentum + adv_grad / torch.mean(torch.abs(adv_grad), dim=(1,2,3), keepdim=True)
            momentum = grad.detach()
            adv_delta = adv_delta + self.alpha*grad.sign()
            adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps).detach()
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images

class DIFGSM(Attack):
    r"""
    DI2-FGSM in the paper 'Improving Transferability of Adversarial Examples with Input Diversity'
    [https://arxiv.org/abs/1803.06978]
    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 0.0)
        steps (int): number of iterations. (Default: 10)
        resize_rate (float): resize factor used in input diversity. (Default: 0.9)
        diversity_prob (float) : the probability of applying input diversity. (Default: 0.5)
        random_start (bool): using random initialization of delta. (Default: False)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = torchattacks.DIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=0.0, resize_rate=0.9, diversity_prob=0.5, random_start=False)
        >>> adv_images = attack(images, labels)
    """

    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, decay=1.0,
                 resize_rate=0.9, diversity_prob=0.7, random_start=False,loss=nn.CrossEntropyLoss(),targeted=False, target=None, normalize=True):
        super().__init__("DIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t
    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # if self.targeted:
        #     labels = self.target*torch.ones_like(labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_delta = adv_delta + torch.empty_like(adv_delta).uniform_(-self.eps, self.eps)
            clean_images = torch.clamp(clean_images+adv_delta, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_delta.requires_grad = True
            adv_images = torch.clamp(clean_images+adv_delta, min=0, max=1)
            
            if self.normalize:
                outputs = self.model(self.norm(self.input_diversity(adv_images)))
            else:
                outputs = self.model(self.input_diversity(adv_images))

            # Calculate loss
            if self.targeted:
                cost = -self.loss(outputs, labels)
            else:
                cost = self.loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_delta,
                                       retain_graph=False, create_graph=False)[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_delta = adv_delta.detach() + self.alpha*grad.sign()
            adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images


class HSFGSM(Attack):

    def __init__(self, model,targets,target_sets, eps=8/255, alpha=2/255, steps=5, decay=1.0,loss=nn.KLDivLoss(reduction='batchmean'),targeted=False, target=None, normalize=True):
        super().__init__("HSFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.target_sets = target_sets
        self.targets = targets
        self.match_dict = self.match_sets()
        self.normalize = normalize
    def match_sets(self):
        match_dict = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
        criterion = nn.CrossEntropyLoss()
        for i in range(len(match_dict)):
            max_grad_norm = 100000
            max_grad_sample = 0.0
            
            for sample in self.target_sets[i]:
                input = Variable(sample[0], requires_grad=True).to(self.device)
                output = self.model(input.unsqueeze(0))
                pred = output.argmax(dim=1)
                sample_loss = criterion(output,torch.tensor(self.targets[i]).unsqueeze(0).to(self.device))
                sample_grad = torch.autograd.grad(sample_loss, input,
                                       retain_graph=False, create_graph=False)[0]
                sample_grad_norm = torch.norm(sample_grad).item()*sample_loss.item()
                if sample_grad_norm < max_grad_norm and pred.item()==self.targets[i]:
                    #max_grad_sample = sample[0]
                    max_grad_sample = output.squeeze(0)
                    max_grad_norm = sample_grad_norm
            match_dict[i].append(max_grad_sample)
        return match_dict
    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # if self.targeted:
            
        #     labels = (self.targets.index(self.target)*torch.ones((images.shape[0],))).long().to(self.device)
        
        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = self.loss

        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()
        for _ in range(self.steps):
            x_match = []
            for i in range(labels.shape[0]):
                #x_match.append(random.choice(self.target_sets[labels[i].item()])[0])
                x_match.append((self.match_dict[labels[i].item()])[0])
            x_match = torch.tensor(np.array([item.cpu().detach().numpy() for item in x_match])).to(self.device)
            target_images = x_match.clone().detach()
            adv_delta.requires_grad = True
            adv_images = torch.clamp(clean_images+adv_delta, min=0, max=1)
            target_outputs = x_match.clone().detach()
            if self.normalize:
                outputs = self.model(self.norm(adv_images))
            else:
                outputs = self.model(adv_images)
            # Calculate loss
            if self.targeted:
                target_out = F.log_softmax(target_outputs,dim=-1)
                outputs = F.softmax(outputs,dim=-1)
                cost = -loss(target_out,outputs)
                #print(cost.item())
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_delta,
                                       retain_graph=False, create_graph=False)[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_delta = adv_delta.detach() + self.alpha*grad.sign()
            adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images

class HSPGD(Attack):
    
    def __init__(self, model,targets,target_sets, random_start=True,eps=8/255, alpha=2/255, steps=5, decay=1.0,loss=nn.KLDivLoss(reduction='batchmean'),targeted=False, target=None, normalize=True):
        super().__init__("HSPGD", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.target_sets = target_sets
        self.targets = targets
        self.random_start = random_start
        self.match_dict = self.match_sets()
        self.normalize = normalize
    def match_sets(self):
        match_dict = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
        criterion = nn.CrossEntropyLoss()
        for i in range(len(match_dict)):
            max_grad_norm = 100000
            max_grad_sample = 0.0
            
            for sample in self.target_sets[i]:
                input = Variable(sample[0], requires_grad=True).to(self.device)
                output = self.model(input.unsqueeze(0))
                pred = output.argmax(dim=1)
                sample_loss = criterion(output,torch.tensor(self.targets[i]).unsqueeze(0).to(self.device))
                sample_grad = torch.autograd.grad(sample_loss, input,
                                       retain_graph=False, create_graph=False)[0]
                sample_grad_norm = torch.norm(sample_grad).item()
                if sample_grad_norm < max_grad_norm and pred.item()==self.targets[i]:
                    #max_grad_sample = sample[0]
                    max_grad_sample = output.squeeze(0)
                    max_grad_norm = sample_grad_norm
            match_dict[i].append(max_grad_sample)
        return match_dict
    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        loss = self.loss
        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()
        if self.random_start:
            # Starting at a uniformly random point
            adv_delta = adv_delta + torch.empty_like(adv_delta).uniform_(-self.eps, self.eps)
            clean_images = torch.clamp(clean_images+adv_delta, min=0, max=1).detach()

        for _ in range(self.steps):

            if self.targeted:
            
                labels = (self.targets.index(self.target)*torch.ones((images.shape[0],))).long().to(self.device)
            x_match = []
            for i in range(labels.shape[0]):
                #x_match.append((self.target_sets[labels[i].item()])[0][0])
                #x_match.append(random.choice(self.target_sets[labels[i].item()])[0])
                x_match.append((self.match_dict[labels[i].item()])[0])
            x_match = torch.tensor(np.array([item.cpu().detach().numpy() for item in x_match])).to(self.device)
            target_outputs = x_match.clone().detach()

            adv_delta.requires_grad = True
            adv_images = torch.clamp(clean_images+adv_delta, min=0, max=1)
            if self.normalize:
                outputs = self.model(self.norm(adv_images))
            else:
                outputs = self.model(adv_images)
            # Calculate loss
            if self.targeted:
                target_out = F.log_softmax(target_outputs,dim=-1)
                outputs = F.softmax(outputs,dim=-1)
                cost = -loss(target_out,outputs)
                #print(cost.item())
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_delta,
                                       retain_graph=False, create_graph=False)[0]

            adv_delta = adv_delta.detach() + self.alpha*grad.sign()
            adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images
    

class Bo_TIFGSM(Attack):
    
    def __init__(self, model, eps=8/255, alpha=2/255, steps=20, decay=1.0, kernel_name='gaussian',
                 len_kernel=5, nsig=3, resize_rate=0.9, diversity_prob=0.7, p=0.5, random_start=False,
                 loss=nn.CrossEntropyLoss(),targeted=False, target=None,normalize=True):
        super().__init__("Bo_TIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())
        self.p = p
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t
    def double_kl_div(self, p_output, q_output, get_softmax=True, gamma=0.01):

        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        
        return gamma*KLDivLoss(F.log_softmax(p_output,dim=-1), F.softmax(q_output,dim=-1))+KLDivLoss(F.log_softmax(q_output,dim=-1), F.softmax(p_output,dim=-1))
    def random_region_mask(self, input_tensor, region_size=128, mask_ratio=0.3):
        output_tensor = input_tensor.clone()  # 克隆输入张量以保留原始数据
        b, c, h, w = input_tensor.shape
        # 随机选择遮盖区域的起始位置
        x = random.randint(0, h - region_size)
        y = random.randint(0, w - region_size)

        # 计算要遮盖的元素数量
        num_elements = int(region_size * region_size * mask_ratio)

        # 获取随机遮盖的元素索引
        indices = random.sample(range(region_size * region_size), num_elements)

        # 将随机选取的元素置零
        for idx in indices:
            i = idx // region_size
            j = idx % region_size
            output_tensor[:, :, x+i, y+j] = 0

        return output_tensor
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            labels = self.target*torch.ones_like(labels)

        loss = self.loss
        momentum = torch.zeros_like(images).detach().to(self.device)
        stacked_kernel = self.stacked_kernel.to(self.device)

        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()
        #mask = torch.bernoulli(torch.full_like(clean_images, self.p))
        #clean_images_masked = clean_images*mask
        #clean_images_masked = self.random_region_mask(clean_images)
        clean_images_masked = clean_images+0.1*(torch.randn_like(clean_images))
        clean_images_masked = torch.clamp(clean_images_masked, min=0, max=1)
        if self.random_start:
            # Starting at a uniformly random point
            adv_delta = adv_delta + torch.empty_like(adv_delta).uniform_(-self.eps, self.eps)
            clean_images = torch.clamp(clean_images+adv_delta, min=0, max=1).detach()

        for _ in range(self.steps):
            #adv_images.requires_grad = True
            
            adv_delta.requires_grad = True
            adv_images = torch.clamp(clean_images+adv_delta, min=0, max=1)
            adv_images_masked = torch.clamp(clean_images_masked+adv_delta, min=0, max=1)
            if self.normalize:
                outputs = self.model(self.norm(self.input_diversity(adv_images)))
                outputs_masked = self.model(self.norm(self.input_diversity(adv_images_masked)))
            else:
                outputs = self.model(self.input_diversity(adv_images))
                outputs_masked = self.model(self.input_diversity(adv_images_masked))

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, labels)-0.1*loss(outputs_masked, labels)-0.01*self.double_kl_div(outputs, outputs_masked,gamma=0.1)
                #cost = -loss(outputs, labels)-0.1*self.double_kl_div(outputs, outputs_masked)
            else:
                cost = loss(outputs, labels)+0.1*loss(outputs_masked, labels)-0.01*self.double_kl_div(outputs, outputs_masked,gamma=0.1)
                #cost = loss(outputs, labels)-0.1*self.double_kl_div(outputs, outputs_masked)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_delta,
                                       retain_graph=False, create_graph=False)[0]
            # depth wise conv2d
            grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_delta = adv_delta.detach() + self.alpha*grad.sign()
            adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x



class Po_TI_Trip_FGSM(Attack):

    def __init__(self, model, eps=8/255, alpha=2/255, steps=20, decay=1.0, kernel_name='gaussian',
                 len_kernel=5, nsig=3, resize_rate=0.9, diversity_prob=0.7, random_start=False,
                 targeted=False, target=None,normalize = True):
        super().__init__("TIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())
        
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t
    def delta(self,u,v):
        
        delta = 2*(torch.sum((u-v)**2,dim=1,keepdim=True)/((1-torch.sum(u**2,dim=1,keepdim=True))*(1-torch.sum(v**2,dim=1,keepdim=True))))
        return delta
    
    def poincare_distance(self,outputs,target_labels):
        
        target_labels = F.one_hot(target_labels,num_classes=1000).float()
        outputs_norm_l1 = torch.norm(outputs,p=1,dim=1,keepdim=True)
        u = outputs/outputs_norm_l1
        v = torch.clamp(target_labels, min=0, max=0.99999)
        delta = self.delta(u,v)
        poincare = torch.arccosh(1+delta)
        return poincare.squeeze(1)
    
    def triplet_distance(self,outputs,labels,target_labels):
        
        target_labels = F.one_hot(target_labels,num_classes=1000).float()
        labels = F.one_hot(labels,num_classes=1000).float()
        D_tar = 1-torch.abs(torch.diag(torch.matmul(outputs,target_labels.t())))/(torch.norm(outputs,p=2,dim=1)*torch.norm(target_labels,p=2,dim=1))
        D_true = 1-torch.abs(torch.diag(torch.matmul(outputs,labels.t())))/(torch.norm(outputs,p=2,dim=1)*torch.norm(labels,p=2,dim=1))
        gamma = 0.007
        triplet_loss = torch.clamp(D_tar - D_true + gamma, min=0, max=1)
        return triplet_loss
    
    def po_trip(self,outputs,labels,target_labels):
        #print(self.poincare_distance(outputs,labels).shape,self.triplet_distance(outputs, labels, target_labels).shape)
        po_trip_loss = self.poincare_distance(outputs,target_labels)+0.01*self.triplet_distance(outputs, labels, target_labels)
        #print(po_trip_loss.shape)
        return po_trip_loss.mean()
    
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = labels

        #loss = self.loss
        momentum = torch.zeros_like(images).detach().to(self.device)
        stacked_kernel = self.stacked_kernel.to(self.device)

        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_delta = adv_delta + torch.empty_like(adv_delta).uniform_(-self.eps, self.eps)
            clean_images = torch.clamp(clean_images+adv_delta, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_delta.requires_grad = True
            adv_images = torch.clamp(clean_images+adv_delta, min=0, max=1)
            if self.normalize:
                outputs = self.model(self.norm(self.input_diversity(adv_images)))
            else:
                outputs = self.model(self.input_diversity(adv_images))

            # Calculate loss
            if self.targeted:
                cost = -self.po_trip(outputs, labels, target_labels)
            else:
                cost = self.po_trip(outputs, labels, target_labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_delta,
                                       retain_graph=False, create_graph=False)[0]
            # depth wise conv2d
            grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_delta = adv_delta.detach() + self.alpha*grad.sign()
            adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x
    

class VMIFGSM(Attack):
    r'''
    variance tuning NIFGSM
    '''
    def __init__(self, model, eps=16/255, steps=10, decay=1.0,loss=nn.CrossEntropyLoss(),\
                    targeted=False, target=None, normalize = True,max_iter=20,beta=1.5):
        super().__init__("VMIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = eps / steps
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
        self.max_iter = max_iter
        self.beta = beta

    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t
    def Var(self, images, labels):
        '''
        本函数用于近似计算方差过程中,求取x邻域的导数
        images:原本的图片
        labels:原本的类别label
        '''
        # print(self.model.__class__.__name__)
        grad = 0
        loss = self.loss
        alpha = self.eps*self.beta
        for i in range(self.max_iter):
            #minval=-alpha, maxval=alpha
            images_r = torch.rand(images.shape,device=self.device)*2*alpha-alpha
            images_r.detach_()
            images_r.requires_grad = True
            images_neighbor = images + images_r
            if self.normalize:
                normalized_adv_images = self.norm(images_neighbor)
                outputs = self.model(normalized_adv_images)
            else:
                outputs = self.model(images_neighbor)
            # outputs = self.model(images_neighbor)
            if self.targeted:
                cost = -loss(outputs, labels)
            else:
                cost = loss(outputs, labels)
            grad += torch.autograd.grad(cost, images_r,
                                       retain_graph=False, create_graph=False)[0]
        var = grad / (1. * self.max_iter)
        return var
    
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        # print(self.model.__class__.__name__)
        var_images = images.clone().to(self.device)
        var_labels = labels.clone().to(self.device)
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # if self.targeted:
        #     labels = self.target*torch.ones_like(labels)

        momentum = torch.zeros_like(images).detach().to(self.device)
        old_var = torch.zeros_like(images).detach().to(self.device)

        loss = self.loss

        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()

        for _ in range(self.steps):
            #adv_images.requires_grad = True
            adv_delta.requires_grad = True
            adv_images = torch.clamp(clean_images+adv_delta, min=0, max=1)
            if self.normalize:
                normalized_adv_images = self.norm(adv_images)
                outputs = self.model(normalized_adv_images)
            else:
                outputs = self.model(adv_images)
            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, labels)
            else:
                cost = loss(outputs, labels)
            
            # Update adversarial images
            new_grad = torch.autograd.grad(cost, adv_delta,
                                       retain_graph=False, create_graph=False)[0]
            # variance tuning
            #! add the var to grad
            var = self.Var(var_images,var_labels) - new_grad
            grad = new_grad + old_var
            #! update old var
            old_var = var

            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_delta = adv_delta.detach() + self.alpha*grad.sign()
            adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images

class VNIFGSM(Attack):
    r'''
    variance tuning NIFGSM
    '''
    def __init__(self, model, eps=16/255, steps=10, decay=1.0,loss=nn.CrossEntropyLoss(),\
                 targeted=False, target=None, normalize = True,max_iter=20,beta=1.5):
        super().__init__("VNIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = eps / steps
        self.supported_mode = ['default', 'targeted']
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
        self.max_iter = max_iter
        self.beta = beta
        
    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]
        return t
    
    def Var(self, images, labels):
        '''
        本函数用于近似计算方差过程中,求取x邻域的导数
        images:原本的图片
        labels:原本的类别label
        '''
        # print(self.model.__class__.__name__)
        grad = 0
        loss = self.loss
        alpha = self.eps*self.beta
        for i in range(self.max_iter):
            # minval=-alpha, maxval=alpha
            images_r = torch.rand(images.shape,device=self.device)*2*alpha-alpha
            images_r.detach_()
            images_r.requires_grad = True
            images_neighbor = images + images_r
            if self.normalize:
                normalized_adv_images = self.norm(images_neighbor)
                outputs = self.model(normalized_adv_images)
            else:
                outputs = self.model(images_neighbor)
            # outputs = self.model(images_neighbor)
            # cost = loss(outputs, labels)
            if self.targeted:
                cost = -loss(outputs, labels)
            else:
                cost = loss(outputs, labels)
            grad += torch.autograd.grad(cost, images_r,
                                       retain_graph=False, create_graph=False)[0]
        var = grad / (1. * self.max_iter)
        return var
    
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        var_images = images.clone().to(self.device)
        var_labels = labels.clone().to(self.device)
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # if self.targeted:
        #     labels = self.target*torch.ones_like(labels)

        momentum = torch.zeros_like(images).detach().to(self.device)
        old_var = torch.zeros_like(images).detach().to(self.device)
        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()

        for _ in range(self.steps):
            #adv_images.requires_grad = True
            adv_delta.requires_grad = True
            adv_images = clean_images+adv_delta
            nes_images = torch.clamp(adv_images + self.decay*self.alpha*momentum, min=0, max=1)
            if self.normalize:
                outputs = self.model(self.norm(nes_images))
            else:
                outputs = self.model(nes_images)
            # Calculate loss
            if self.targeted:
                cost = -self.loss(outputs, labels)
            else:
                cost = self.loss(outputs, labels)

            # Update adversarial images
            new_grad = torch.autograd.grad(cost, adv_delta,
                                       retain_graph=False, create_graph=False)[0]
            
            # variance tuning
            #! add the var to grad
            var = self.Var(var_images,var_labels) - new_grad
            grad = new_grad + old_var
            #! update old var
            old_var = var
            
            grad = self.decay*momentum + grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            momentum = grad
            adv_delta = adv_delta.detach() + self.alpha*grad.sign()
            adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images

class VTI_DI_MIFGSM(Attack):
    def __init__(self, model, eps=16/255, steps=10, decay=1.0, kernel_name='gaussian',
                 len_kernel=5, nsig=3, resize_rate=0.9, diversity_prob=0.7, random_start=False,
                 loss=nn.CrossEntropyLoss(),targeted=False, target=None,normalize=True,
                 max_iter=20,beta=1.5):
        super().__init__("VTI_DI_MIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = eps / steps
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
        self.max_iter = max_iter
        self.beta = beta
    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t
    def Var(self, images, labels):
        '''
        本函数用于近似计算方差过程中,求取x邻域的导数
        images:原本的图片
        labels:原本的类别label
        '''
        # print(self.model.__class__.__name__)
        grad = 0
        loss = self.loss
        alpha = self.eps*self.beta
        for i in range(self.max_iter):
            # minval=-alpha, maxval=alpha
            images_r = torch.rand(images.shape,device=self.device)*2*alpha-alpha
            images_r.detach_()
            images_r.requires_grad = True
            images_neighbor = images + images_r
            if self.normalize:
                normalized_adv_images = self.norm(images_neighbor)
                outputs = self.model(normalized_adv_images)
            else:
                outputs = self.model(images_neighbor)
            # outputs = self.model(images_neighbor)
            # cost = loss(outputs, labels)
            if self.targeted:
                cost = -loss(outputs, labels)
            else:
                cost = loss(outputs, labels)
            grad += torch.autograd.grad(cost, images_r,
                                       retain_graph=False, create_graph=False)[0]
        var = grad / (1. * self.max_iter)
        return var
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        var_images = images.clone().to(self.device)
        var_labels = labels.clone().to(self.device)
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # if self.targeted:
        #     labels = self.target*torch.ones_like(labels)

        loss = self.loss
        momentum = torch.zeros_like(images).detach().to(self.device)
        old_var = torch.zeros_like(images).detach().to(self.device)
        stacked_kernel = self.stacked_kernel.to(self.device)

        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_delta = adv_delta + torch.empty_like(adv_delta).uniform_(-self.eps, self.eps)
            clean_images = torch.clamp(clean_images+adv_delta, min=0, max=1).detach()

        for _ in range(self.steps):
            #adv_images.requires_grad = True
            adv_delta.requires_grad = True
            adv_images = torch.clamp(clean_images+adv_delta, min=0, max=1)
            if self.normalize:
                outputs = self.model(self.norm(self.input_diversity(adv_images)))
            else:
                outputs = self.model(self.input_diversity(adv_images))

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            new_grad = torch.autograd.grad(cost, adv_delta,
                                       retain_graph=False, create_graph=False)[0]
            # variance tuning
            #! add the var to grad
            var = self.Var(var_images,var_labels) - new_grad
            grad = new_grad + old_var
            #! update old var
            old_var = var
            # depth wise conv2d
            grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)
            
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_delta = adv_delta.detach() + self.alpha*grad.sign()
            adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x
    
class VTI_DI_SI_MIFGSM(Attack):#!其实是NI
    def __init__(self, model, eps=16/255, steps=10, decay=1.0, kernel_name='gaussian',
                 len_kernel=5, nsig=3, resize_rate=0.9, diversity_prob=0.7, random_start=False,
                 loss=nn.CrossEntropyLoss(),targeted=False, target=None,normalize=True,
                 max_iter=20,beta=1.5,m=5):
        super().__init__("VTI_DI_SI_MIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = eps / steps
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
        self.max_iter = max_iter
        self.beta = beta
        self.m = m
    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t
    def Var(self, images, labels):
        '''
        本函数用于近似计算方差过程中,求取x邻域的导数
        images:原本的图片
        labels:原本的类别label
        '''
        # print(self.model.__class__.__name__)
        loss = self.loss
        grad = 0
        alpha = self.eps*self.beta
        for i in range(self.max_iter):
            # minval=-alpha, maxval=alpha
            images_r = torch.rand(images.shape,device=self.device)*2*alpha-alpha
            images_r.detach_()
            images_r.requires_grad = True
            images_neighbor = images + images_r
            adv_grad = torch.zeros_like(images).detach().to(self.device)
            for i in torch.arange(self.m):
                nes_images = images_neighbor / torch.pow(2, i)
                if self.normalize:
                    outputs = self.model(self.norm(self.input_diversity(nes_images)))
                else:
                    outputs = self.model(self.input_diversity(nes_images))
                # Calculate loss
                if self.targeted:
                    cost = -loss(outputs, labels)
                else:
                    cost = loss(outputs, labels)
                adv_grad += torch.autograd.grad(cost, images_r,
                                                retain_graph=True, create_graph=True)[0].detach()
            adv_grad = adv_grad / self.m
            grad += adv_grad
        var = grad / (1. * self.max_iter)
        return var
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        var_images = images.clone().to(self.device)
        var_labels = labels.clone().to(self.device)
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # if self.targeted:
        #     labels = self.target*torch.ones_like(labels)

        loss = self.loss
        momentum = torch.zeros_like(images).detach().to(self.device)
        old_var = torch.zeros_like(images).detach().to(self.device)
        stacked_kernel = self.stacked_kernel.to(self.device)

        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_delta = adv_delta + torch.empty_like(adv_delta).uniform_(-self.eps, self.eps)
            clean_images = torch.clamp(clean_images+adv_delta, min=0, max=1).detach()

        # Calculate sum the gradients over the scale copies of the input image
        for _ in range(self.steps):
            adv_delta.requires_grad = True
            adv_images = clean_images+adv_delta
            nes_images = torch.clamp(adv_images + self.decay*self.alpha*momentum, min=0, max=1)
    
            # Calculate sum the gradients over the scale copies of the input image
            new_grad = torch.zeros_like(images).detach().to(self.device)
            for i in torch.arange(self.m):
                nes_images = nes_images / torch.pow(2, i)
                if self.normalize:
                    outputs = self.model(self.norm(self.input_diversity(nes_images)))
                else:
                    outputs = self.model(self.input_diversity(nes_images))
                # Calculate loss
                if self.targeted:
                    cost = -self.loss(outputs, labels)
                else:
                    cost = self.loss(outputs, labels)
                new_grad += torch.autograd.grad(cost, adv_delta,
                                                retain_graph=True, create_graph=True)[0].detach()

            # Update adversarial images
            # new_grad = torch.autograd.grad(cost, adv_delta,
            #                            retain_graph=False, create_graph=False)[0]
            new_grad = new_grad / self.m
            # variance tuning
            #! add the var to grad
            var = self.Var(var_images,var_labels) - new_grad
            grad = new_grad + old_var
            #! update old var
            old_var = var
            # depth wise conv2d
            grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)
            
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_delta = adv_delta.detach() + self.alpha*grad.sign()
            adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x
    
class VNIFGSM(Attack):
    r'''
    variance tuning NIFGSM
    '''
    def __init__(self, model, eps=16/255, steps=10, decay=1.0,loss=nn.CrossEntropyLoss(),\
                 targeted=False, target=None, normalize = True,max_iter=20,beta=1.5):
        super().__init__("VNIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = eps / steps
        self.supported_mode = ['default', 'targeted']
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
        self.max_iter = max_iter
        self.beta = beta
        
    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]
        return t
    
    def Var(self, images, labels):
        '''
        本函数用于近似计算方差过程中,求取x邻域的导数
        images:原本的图片
        labels:原本的类别label
        '''
        # print(self.model.__class__.__name__)
        grad = 0
        loss = self.loss
        alpha = self.eps*self.beta
        for i in range(self.max_iter):
            # minval=-alpha, maxval=alpha
            images_r = torch.rand(images.shape,device=self.device)*2*alpha-alpha
            images_r.detach_()
            images_r.requires_grad = True
            images_neighbor = images + images_r
            if self.normalize:
                normalized_adv_images = self.norm(images_neighbor)
                outputs = self.model(normalized_adv_images)
            else:
                outputs = self.model(images_neighbor)
            # outputs = self.model(images_neighbor)
            cost = loss(outputs, labels)
            grad += torch.autograd.grad(cost, images_r,
                                       retain_graph=False, create_graph=False)[0]
        var = grad / (1. * self.max_iter)
        return var
    
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        var_images = images.clone().to(self.device)
        var_labels = labels.clone().to(self.device)
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # if self.targeted:
        #     labels = self.target*torch.ones_like(labels)

        momentum = torch.zeros_like(images).detach().to(self.device)
        old_var = torch.zeros_like(images).detach().to(self.device)
        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()

        for _ in range(self.steps):
            #adv_images.requires_grad = True
            adv_delta.requires_grad = True
            adv_images = clean_images+adv_delta
            nes_images = torch.clamp(adv_images + self.decay*self.alpha*momentum, min=0, max=1)
            if self.normalize:
                outputs = self.model(self.norm(nes_images))
            else:
                outputs = self.model(nes_images)
            # Calculate loss
            if self.targeted:
                cost = -self.loss(outputs, labels)
            else:
                cost = self.loss(outputs, labels)

            # Update adversarial images
            new_grad = torch.autograd.grad(cost, adv_delta,
                                       retain_graph=False, create_graph=False)[0]
            
            # variance tuning
            #! add the var to grad
            var = self.Var(var_images,var_labels) - new_grad
            grad = new_grad + old_var
            #! update old var
            old_var = var
            
            grad = self.decay*momentum + grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            momentum = grad
            adv_delta = adv_delta.detach() + self.alpha*grad.sign()
            adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images
    
class Logits(Attack):

    def __init__(self, model, eps=8/255, alpha=2/255, steps=20, decay=1.0, random_start=False,
                 loss=nn.CrossEntropyLoss(),targeted=False, target=None,normalize=True):
        super().__init__("Logits", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.random_start = random_start
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize

    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t
    
    def logit_loss(self, out, labels):
        return -torch.gather(out, 1, labels.view(-1, 1)).squeeze(-1).mean()

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # if self.targeted:
        #     labels = self.target*torch.ones_like(labels)

        #loss = self.loss
        momentum = torch.zeros_like(images).detach().to(self.device)
        # stacked_kernel = self.stacked_kernel.to(self.device)

        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_delta = adv_delta + torch.empty_like(adv_delta).uniform_(-self.eps, self.eps)
            clean_images = torch.clamp(clean_images+adv_delta, min=0, max=1).detach()

        for _ in range(self.steps):
            #adv_images.requires_grad = True
            adv_delta.requires_grad = True
            adv_images = torch.clamp(clean_images+adv_delta, min=0, max=1)
            if self.normalize:
                # outputs = self.model(self.norm(self.input_diversity(adv_images)))
                outputs = self.model(self.norm(adv_images))
            else:
                # outputs = self.model(self.input_diversity(adv_images))
                outputs = self.model(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -self.logit_loss(outputs, labels)
            else:
                cost = self.logit_loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_delta,
                                       retain_graph=False, create_graph=False)[0]
            # depth wise conv2d
            # grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_delta = adv_delta.detach() + self.alpha*grad.sign()
            adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images
    
class Logits_NI(Attack):

    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, decay=1.0,loss=nn.CrossEntropyLoss(),targeted=False, target=None, normalize = True):
        super().__init__("Logits_NI", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        #! self.alpha = eps / steps
        self.supported_mode = ['default', 'targeted']
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t
    
    def logit_loss(self, out, labels):
        return -torch.gather(out, 1, labels.view(-1, 1)).squeeze(-1).mean()
    
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # if self.targeted:
        #     labels = self.target*torch.ones_like(labels)

        momentum = torch.zeros_like(images).detach().to(self.device)
        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()

        for _ in range(self.steps):
            #adv_images.requires_grad = True
            adv_delta.requires_grad = True
            adv_images = clean_images+adv_delta
            nes_images = torch.clamp(adv_images + self.decay*self.alpha*momentum, min=0, max=1)
            if self.normalize:
                outputs = self.model(self.norm(nes_images))
            else:
                outputs = self.model(nes_images)
            # Calculate loss
            if self.targeted:
                cost = -self.logit_loss(outputs, labels)
            else:
                cost = self.logit_loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_delta,
                                       retain_graph=False, create_graph=False)[0]
            grad = self.decay*momentum + grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            momentum = grad
            adv_delta = adv_delta.detach() + self.alpha*grad.sign()
            adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images
    
class Logits_Trace_NI(Attack):

    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, decay=1.0,loss=nn.CrossEntropyLoss(),\
                    targeted=False, target=None, normalize = True,num_samples=20,lamda_trace=0.5):
        super().__init__("Logits_Trace_NI", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        #! self.alpha = eps / steps
        self.supported_mode = ['default', 'targeted']
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
        #! for trace
        self.num_samples = num_samples
        self.lamda_trace = lamda_trace
    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t
    
    def logit_loss(self, out, labels):
        return -torch.gather(out, 1, labels.view(-1, 1)).squeeze(-1).mean()
    
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # if self.targeted:
        #     labels = self.target*torch.ones_like(labels)

        momentum = torch.zeros_like(images).detach().to(self.device)
        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()

        for _ in range(self.steps):
            #adv_images.requires_grad = True
            adv_delta.requires_grad = True
            adv_images = clean_images+adv_delta
            nes_images = torch.clamp(adv_images + self.decay*self.alpha*momentum, min=0, max=1)
            if self.normalize:
                outputs = self.model(self.norm(nes_images))
            else:
                outputs = self.model(nes_images)
            # calculate trace
            #! 计算每一张图片的hessian矩阵的迹
            trace = []
            # torch.cuda.synchronize()   #增加同步操作
            # start = time.time()
            for i in range(nes_images.shape[0]):
                current_trace = hutchinson_trace_hvp(self.num_samples,nes_images[i],\
                                    self.logit_loss,self.model,labels[i],self.device)
                # print(current_trace)
                trace.append(current_trace)
            # torch.cuda.synchronize() #增加同步操作
            # end = time.time()
            # print(f'trace time: {end-start}')
            trace = torch.cat(trace)
            # Calculate loss
            if self.targeted:
                cost = -self.logit_loss(outputs, labels) + self.lamda_trace * trace.mean()
                # print(cost)
            else:
                cost = self.logit_loss(outputs, labels) + self.lamda_trace * trace.mean()

            # Update adversarial images
            # torch.cuda.synchronize()   #增加同步操作
            # start = time.time()
            grad = torch.autograd.grad(cost, adv_delta,
                                       retain_graph=False, create_graph=False)[0]
            # torch.cuda.synchronize() #增加同步操作
            # end = time.time()
            # print(f'grad time: {end-start}')
            grad = self.decay*momentum + grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            momentum = grad
            adv_delta = adv_delta.detach() + self.alpha*grad.sign()
            adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images
    
class NI_trace(Attack):

    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, decay=1.0,loss=nn.CrossEntropyLoss(),\
                 targeted=False, target=None, normalize = True,num_samples=20,lamda_trace=0.5):
        super().__init__("NI_trace", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        #! self.alpha = eps / steps
        self.supported_mode = ['default', 'targeted']
        self.loss = loss
        self.targeted = targeted
        self.target = target
        self.normalize = normalize
        #! for trace
        self.num_samples = num_samples
        self.lamda_trace = lamda_trace
    def norm(self, t):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

        return t
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # if self.targeted:
        #     labels = self.target*torch.ones_like(labels)

        momentum = torch.zeros_like(images).detach().to(self.device)
        adv_delta = torch.zeros_like(images).to(self.device)
        
        clean_images = images.clone().detach()

        for _ in range(self.steps):
            #adv_images.requires_grad = True
            adv_delta.requires_grad = True
            adv_images = clean_images+adv_delta
            nes_images = torch.clamp(adv_images + self.decay*self.alpha*momentum, min=0, max=1)
            if self.normalize:
                outputs = self.model(self.norm(nes_images))
            else:
                outputs = self.model(nes_images)
            # calculate trace
            #! 计算每一张图片的hessian矩阵的迹
            trace = []
            for i in range(nes_images.shape[0]):
                current_trace = hutchinson_trace_hvp(self.num_samples,nes_images[i],\
                                    self.loss,self.model,labels[i],self.device)
                trace.append(current_trace)
            trace = torch.cat(trace)
            # Calculate loss
            if self.targeted:
                cost = -self.loss(outputs, labels) + self.lamda_trace * trace.mean()
            else:
                cost = self.loss(outputs, labels) + self.lamda_trace * trace.mean()

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_delta,
                                       retain_graph=False, create_graph=False)[0]
            grad = self.decay*momentum + grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            momentum = grad
            adv_delta = adv_delta.detach() + self.alpha*grad.sign()
            adv_delta = torch.clamp(adv_delta, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + adv_delta, min=0, max=1).detach()

        return adv_images