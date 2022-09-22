import torch
import torch.nn as nn
import os
import argparse
import logging
from torch.autograd import Variable
from torch.utils.data import DataLoader
from segDataset import segDataset
from evaluation import test_current_model
from Visualizer import Visualizer
from vit_seg_modeling import VisionTransformer as ViT_seg
from vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import numpy as np
import torch

# Parameters
betas = (0.9, 0.999)
file_path = '/root/data/2_TransUNet-main/TransUNet-main/percy'

def getArgs():
    parse = argparse.ArgumentParser(description='PyTorch Segmentation')
    parse.add_argument('--num_classes', type=int,
                        default=1, help='output channel of network')
    parse.add_argument('--base_lr', type=float,  default=0.0001,
                        help='segmentation network learning rate')
    parse.add_argument('--img_size', type=int,
                        default=512, help='input patch size of network input')
    parse.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parse.add_argument('--n_skip', type=int,
                        default=3, help='using number of skip-connect, default is num')
    parse.add_argument('--vit_name', type=str,
                        default='R50-ViT-B_16', help='select one vit model')
    parse.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')
    parse.add_argument("--dataset_path", type=str, default='Mito')  # 数据集路径
    parse.add_argument('--arch', default='TransUnet')
    parse.add_argument("--epochs", type=int, default=1000)
    parse.add_argument("--train_batch_size", type=int, default=4)
    parse.add_argument("--test_batch_size", type=int, default=1)
    parse.add_argument("--log_dir", type=str, default='model_save/log', help="log dir")
    parse.add_argument("--test_model", type=str, default=None, help='path of the model for testing')
    parse.add_argument("--save_path", type=str, default='model_save/')

    args = parse.parse_args()
    return args


# Mito dataset
def getDataset(args):
    train_data_loader = DataLoader(dataset=segDataset(root=file_path, isTraining=True),
                                   batch_size=args.train_batch_size,
                                   shuffle=True)
    test_data_loader = DataLoader(dataset=segDataset(root=file_path, isTraining=False),
                                  batch_size=args.test_batch_size,
                                  shuffle=False)
    return train_data_loader, test_data_loader


# 日志描述
def getLog(args):
    dirname = os.path.join(args.log_dir,'train_batch_size'+str(args.train_batch_size))
    filename = dirname +'/save_log.log'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    logging.basicConfig(
            filename=filename,
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
    return logging


if __name__ == '__main__':
    args = getArgs()
    train_data_loader, test_data_loader = getDataset(args)
    logging = getLog(args)
    logging.info('\n=======\nmodels:%s\nepoch:%s\nbatch size:%s\n========' \
                 %(args.arch, args.epochs, args.train_batch_size))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    G = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(device)
    G.load_from(weights=np.load(config_vit.pretrained_path))
    G = nn.DataParallel(G).to(device)
    # Loss function
    criterion1 = nn.BCELoss().to(device)
    # 优化器
    G_optimizer = torch.optim.Adam(G.parameters(), lr=args.base_lr, betas=betas)
    G_avg_losses = []
    file_name = 'TransUnet'
    vis = Visualizer(env=file_name)
    iouBest = 0.0
    print("********** Training Starting **********")
    for epoch in range(args.epochs):
        G.train()
        G_total_loss = 0.0
        for i, (img, segLabel) in enumerate(train_data_loader):
            img = Variable(img.to(device))
            segLabel = Variable(segLabel.to(device))
            G.zero_grad()
            output = G(img)
            loss_seg = criterion1(output, segLabel)
            G_loss = loss_seg
            G_loss.backward()
            G_optimizer.step()
            G_total_loss += G_loss.item()
            vis.img(name='Input_img', img_=img[0, :, :, :])
            vis.img(name='predSeg', img_=output[0, :, :, :])
            vis.img(name='labelSeg', img_=segLabel[0, :, :, :])
        mean_loss = G_total_loss / len(train_data_loader.dataset)
        vis.plot(name='train_loss', y=mean_loss)
        print('Epoch [%d/%d] || mean_loss: %.4f || learning_rate: %f ' % (epoch + 1, args.epochs, mean_loss,
                                                                          G_optimizer.state_dict()['param_groups'][0][
                                                                              'lr']))
        if (epoch + 1) % 5 == 0:
            print("********** Testing Current Model **********")
            G.eval()
            with torch.no_grad():
                for img, gt in test_data_loader:
                    img = img.cuda()
                    pred = G(img)
                    vis.img(name='Test_img', img_=pred[0, :, :, :])
                    vis.img(name='Test_gt', img_=gt[0, :, :, :])
            loss_arr, auc_arr, acc_arr, sen_arr, fdr_arr, spe_arr, iou_arr, dice_arr = \
            test_current_model(test_data_loader, G, device, criterion=criterion1)
            vis.plot(name='test_IOU', y=iou_arr.mean())
            vis.plot(name='test_DICE', y=dice_arr.mean())
            logging.info("\n********** Testing Current Model **********\nIOU=%.4f DICE=%.4f" %(iou_arr.mean(),dice_arr.mean()))
            save_path0 = args.save_path + args.arch
            isExists = os.path.exists(save_path0)
            if not isExists:
                os.makedirs(save_path0)
            save_path = os.path.join(save_path0, 'net-{}-{}-{}.pth'.format(epoch + 1, args.epochs,str(iou_arr.mean())[0:4]))
            save_path1 = os.path.join(save_path0, 'param-{}-{}-{}.pth'.format(epoch + 1, args.epochs,str(iou_arr.mean())[0:4]))
            if iou_arr.mean() >= iouBest:
                iouBest = iou_arr.mean()
                torch.save(G, save_path)
                torch.save(G.state_dict(), save_path1)
