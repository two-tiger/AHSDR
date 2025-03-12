#!/bin/bash
#SBATCH -J test0310       # 作业名为 test
#SBATCH -o test0310.out   # 屏幕上的输出⽂件重定向到 test.out
#SBATCH -p compute    # 作业提交的分区为 compute
#SBATCH -N 1          # 作业申请 1 个节点
#SBATCH -t 6:00:00    # 任务运⾏的最⻓时间为 1 ⼩时
#SBATCH --gres=gpu:nvidia_rtx_a6000:1 # 申请GPU
#SBATCH -w gpu07      # 指定运⾏作业的节点是 gpu06，若不填写则不指定

python  utils/main.py --model onlinevt --load_best_args --dataset seq-cifar10 --buffer_size 50 --csv_log --num_classes 10 --num_workers 12 --seed 42 --use_attack --buffer_attack MIFGSM --use_l2 --out_dir  ablation/cifar10/l2 >>  inner_step_orgi_attack.log 2>&1