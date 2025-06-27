# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse  # 解析命令行参数
import datetime  # 处理时间信息
import json  # 读写 JSON 文件
import random  # 随机数相关工具
import time  # 时间相关函数
from pathlib import Path  # 文件路径处理

import numpy as np  # 科学计算库
import torch  # PyTorch 主库
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载工具

import datasets  # 数据集相关模块
import util.misc as utils  # 辅助工具函数
from datasets import build_dataset, get_coco_api_from_dataset  # 构建数据集及 COCO 接口
from engine import evaluate, train_one_epoch  # 训练和评估逻辑
from models import build_model  # 构建模型


def get_args_parser():  # 构建命令行参数解析器
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)  # 初始化解析器
    parser.add_argument('--lr', default=1e-4, type=float)  # 主学习率
    parser.add_argument('--lr_backbone', default=1e-5, type=float)  # Backbone 的学习率
    parser.add_argument('--batch_size', default=2, type=int)  # 训练批大小
    parser.add_argument('--weight_decay', default=1e-4, type=float)  # 权重衰减系数
    parser.add_argument('--epochs', default=300, type=int)  # 训练轮数
    parser.add_argument('--lr_drop', default=200, type=int)  # 学习率下降的 epoch
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')  # 梯度裁剪阈值

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")  # 冻结模型权重路径
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")  # 使用的骨干网络名称
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")  # 是否在最后一个 block 使用空洞卷积
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")  # 位置编码类型

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")  # 是否训练分割头

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")  # 是否关闭辅助损失
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")  # 分类匹配权重
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")  # 框位置匹配权重
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")  # giou 匹配权重
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)  # mask 损失权重
    parser.add_argument('--dice_loss_coef', default=1, type=float)  # dice 损失权重
    parser.add_argument('--bbox_loss_coef', default=5, type=float)  # bbox 损失权重
    parser.add_argument('--giou_loss_coef', default=2, type=float)  # giou 损失权重
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")  # 背景类别权重

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')  # 数据集名称
    parser.add_argument('--coco_path', type=str)  # COCO 路径
    parser.add_argument('--coco_panoptic_path', type=str)  # panoptic 数据集路径
    parser.add_argument('--remove_difficult', action='store_true')  # 是否移除困难样本

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')  # 输出目录
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')  # 训练设备
    parser.add_argument('--seed', default=42, type=int)  # 随机种子
    parser.add_argument('--resume', default='', help='resume from checkpoint')  # 恢复训练的检查点
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')  # 起始 epoch
    parser.add_argument('--eval', action='store_true')  # 仅评估模式
    parser.add_argument('--num_workers', default=2, type=int)  # DataLoader 的工作线程数

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')  # 总进程数
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')  # 初始化地址
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)  # 构建模型、损失和后处理器
    model.to(device)  # 将模型移动到目标设备

    model_without_ddp = model  # 在单 GPU 情况下直接使用原模型
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])  # 多卡并行
        model_without_ddp = model.module  # 实际的模型在 module 属性中
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 计算可训练参数量
    print('number of params:', n_parameters)

    param_dicts = [  # 为 backbone 与其他部分设置不同学习率
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)  # AdamW 优化器
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)  # 每隔 lr_drop 个 epoch 下降学习率

    dataset_train = build_dataset(image_set='train', args=args)  # 构建训练集
    dataset_val = build_dataset(image_set='val', args=args)  # 构建验证集

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)  # 分布式采样器
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)  # 随机采样
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)  # 顺序采样

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)  # 训练批采样器

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)  # 训练集加载器
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)  # 验证集加载器

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')  # 读取冻结权重
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)  # 创建输出目录路径
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])  # 加载模型权重
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])  # 恢复优化器状态
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])  # 恢复学习率计划
            args.start_epoch = checkpoint['epoch'] + 1  # 设置起始 epoch

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)  # 仅评估模型
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")  # 正式进入训练循环
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):  # 遍历所有 epoch
        if args.distributed:
            sampler_train.set_epoch(epoch)  # 分布式训练设置 epoch
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)  # 单个 epoch 的训练
        lr_scheduler.step()  # 更新学习率
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )  # 每个 epoch 结束在验证集评测

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}  # 记录日志信息

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")  # 持久化日志

            # 保存评估结果
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))  # 打印总训练时长


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])  # 构建顶级解析器
    args = parser.parse_args()  # 解析参数
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)  # 创建输出目录
    main(args)  # 调用主函数
