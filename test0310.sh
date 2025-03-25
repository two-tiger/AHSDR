#!/bin/bash
#SBATCH -J perturbation_small_select       # 作业名为 test
#SBATCH -o perturbation_small_select.out   # 屏幕上的输出⽂件重定向到 test.out
#SBATCH -p compute    # 作业提交的分区为 compute
#SBATCH -N 1          # 作业申请 1 个节点
#SBATCH -t 6:00:00    # 任务运⾏的最⻓时间为 1 ⼩时
#SBATCH --gres=gpu:nvidia_rtx_a6000:1 # 申请GPU
#SBATCH -w gpu07      # 指定运⾏作业的节点是 gpu06，若不填写则不指定

# python  utils/main.py --model onlinevt --load_best_args --dataset seq-cifar10 --buffer_size 50 --csv_log --num_classes 10 --num_workers 12 --seed 42 --use_perturbation --find_in_step --find_small >>  perturbation_small_select.log 2>&1

python  utils/main.py --model onlinevt --load_best_args --dataset seq-cifar10 --buffer_size 50 --csv_log --num_classes 10 --num_workers 12 --seed 42 --use_grad_diff --find_in_step >> grad_diff_big_sqrt.log 2>&1

# python  utils/main.py --model onlinevt --load_best_args --dataset seq-cifar10 --buffer_size 50 --csv_log --num_classes 10 --num_workers 12 --seed 42 --use_perturbation_diff --find_in_step >> perturbation_grad_diff_big_abs_sqrt.log 2>&1

# python  utils/main.py --model onlinevt --load_best_args --dataset seq-cifar10 --buffer_size 50 --csv_log --num_classes 10 --num_workers 12 --seed 42 --use_grad --find_small --find_in_step >> grad_ema_big.log 2>&1