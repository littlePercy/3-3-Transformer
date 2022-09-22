# -*- coding: utf-8 -*-

import numpy as np
from sklearn import metrics
import cv2
import torch
from PIL import Image
from torchvision import transforms

def extract_mask(pred_arr, gt_arr, mask_arr=None):
    # we want to make them into vectors
    pred_vec = pred_arr.flatten()
    gt_vec = gt_arr.flatten()
    
    if mask_arr is not None:
        mask_vec = mask_arr.flatten()
        idx = list(np.where(mask_vec == 0)[0])
        
        pred_vec = np.delete(pred_vec, idx)
        gt_vec = np.delete(gt_vec, idx)
    
    return pred_vec, gt_vec


def calc_auc(pred_arr, gt_arr, mask_arr=None):
    pred_vec, gt_vec = extract_mask(pred_arr, gt_arr, mask_arr=mask_arr)
    roc_auc = metrics.roc_auc_score(gt_vec.astype('int'), pred_vec)
    # print(roc_auc)
    # print(type(pred_vec), type(gt_vec))
    return roc_auc


def numeric_score(pred_arr, gt_arr):
    """Computation of statistical numerical scores:

    * FP = False Positives
    * FN = False Negatives
    * TP = True Positives
    * TN = True Negatives

    return: tuple (FP, FN, TP, TN)
    """
    FP = np.float(np.sum(np.logical_and(pred_arr == 1, gt_arr == 0)))
    FN = np.float(np.sum(np.logical_and(pred_arr == 0, gt_arr == 1)))
    TP = np.float(np.sum(np.logical_and(pred_arr == 1, gt_arr == 1)))
    TN = np.float(np.sum(np.logical_and(pred_arr == 0, gt_arr == 0)))
    
    return FP, FN, TP, TN

"""
像素准确率Accuracy： 它是图像中正确分类的像素百分比。即分类正确的像素占总像素的比例。
高像素精度并不总是意味着卓越的分割能力=>类别不平衡情况，背景占比极大即使什么也没分出来ACC也很高。
因此，这个指标基本没什么指导意义。
"""
def calc_acc(pred_arr, gt_arr, mask_arr=None):
    pred_vec, gt_vec = extract_mask(pred_arr, gt_arr, mask_arr=mask_arr)
    FP, FN, TP, TN = numeric_score(pred_vec, gt_vec)
    acc = (TP + TN) / (FP + FN + TP + TN)
    
    return acc

"""
灵敏度Sensitive表示所有正例中，被分对的比例，衡量给了分类器对正例的识别能力。
灵敏度与召回率Recall计算公式完全一致。
"""
def calc_sen(pred_arr, gt_arr, mask_arr=None):
    pred_vec, gt_vec = extract_mask(pred_arr, gt_arr, mask_arr=mask_arr)
    FP, FN, TP, TN = numeric_score(pred_vec, gt_vec)
    sen = TP / (FN + TP + 1e-12)
    
    return sen


"""
特效度Specificity表达了对负例的识别能力，和灵敏度相似。
Spe=TN/N=TN/(TN+FP)
"""
def calc_spe(pred_arr, gt_arr, mask_arr=None):
    pred_vec, gt_vec = extract_mask(pred_arr, gt_arr, mask_arr=mask_arr)
    FP, FN, TP, TN = numeric_score(pred_vec, gt_vec)
    spe = TN / (FP + TN + 1e-12)

    return spe

"""
FDR（false discovery rate），错误发现率
其意义为是错误拒绝（拒绝真的（原）假设）的个数占所有被拒绝的原假设个数的比例的期望值。
"""
def calc_fdr(pred_arr, gt_arr, mask_arr=None):
    pred_vec, gt_vec = extract_mask(pred_arr, gt_arr, mask_arr=mask_arr)
    FP, FN, TP, TN = numeric_score(pred_vec, gt_vec)
    fdr = FP / (FP + TP + 1e-12)
    
    return fdr


"""
IoU 是预测分割和标签之间的重叠区域除以预测分割和标签之间的联合区域（两者的交集/两者的并集）。
该指标的范围为 0–1 (0–100%)，其中 0 表示没有重叠，1 表示完全重叠分割。
"""
def calc_iou(pred_arr, gt_arr, mask_arr=None):
    pred_vec, gt_vec = extract_mask(pred_arr, gt_arr, mask_arr=mask_arr)
    FP, FN, TP, TN = numeric_score(pred_vec, gt_vec)
    iou = TP / (FP + FN + TP + 1e-12)
    
    return iou

"""
Dice系数: 定义为两倍的交集除以像素和，也叫F1 score。
Dice 系数与 IoU 非常相似，它们是正相关的。
与 IoU 一样，它们的范围都从 0 到 1，其中 1 表示预测和真实之间的最大相似度。
"""
def calc_dice(pred_arr, gt_arr, mask_arr=None):
    pred_vec, gt_vec = extract_mask(pred_arr, gt_arr, mask_arr=mask_arr)
    FP, FN, TP, TN = numeric_score(pred_vec, gt_vec)
    dice = 2.0 * TP / (FP + FN + 2.0 * TP + 1e-12)
    
    return dice


def test_current_model(dataloader, net, device, criterion=None):
    loss_lst = []  # Loss
    auc_lst = []  # AUC
    acc_lst = []  # Accuracy
    sen_lst = []  # Sensitivity (Recall)
    fdr_lst = []  # FDR
    spe_lst = []  # Specificity
    iou_lst = []  # IOU
    dice_lst = []  # Dice Coefficient (F1-score)

    i = 1
    with torch.no_grad():
        for sample in dataloader:
            i += 1
            img = sample[0].to(device) # 原图
            gt = sample[1].to(device)  # 分割label
            pred = net(img)
            if len(sample) == 5:
                w, h = sample[4]

                pred = pred.cpu().squeeze(0)
                pred = transforms.ToTensor()(transforms.ToPILImage()(pred))
                pred = pred.unsqueeze(0).to(device)

                mask = sample[2].to(device)
                mask_arr = mask.squeeze().cpu().numpy()
            else:
                mask_arr = None

            if criterion is not None:
                loss_lst.append(criterion(pred, gt).item())

            pred_arr = pred.squeeze().cpu().numpy()
            gt_arr = gt.squeeze().cpu().numpy()
            auc_lst.append(calc_auc(pred_arr, gt_arr, mask_arr=mask_arr))

            pred_img = np.array(pred_arr * 255, np.uint8)
            gt_img = np.array(gt_arr * 255, np.uint8)

            thresh_value, thresh_pred_img = cv2.threshold(pred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            acc_lst.append(calc_acc(thresh_pred_img / 255.0, gt_img / 255.0, mask_arr=mask_arr))
            sen_lst.append(calc_sen(thresh_pred_img / 255.0, gt_img / 255.0, mask_arr=mask_arr))
            fdr_lst.append(calc_fdr(thresh_pred_img / 255.0, gt_img / 255.0, mask_arr=mask_arr))
            spe_lst.append(calc_spe(thresh_pred_img / 255.0, gt_img / 255.0, mask_arr=mask_arr))
            iou_lst.append(calc_iou(thresh_pred_img / 255.0, gt_img / 255.0, mask_arr=mask_arr))
            dice_lst.append(calc_dice(thresh_pred_img / 255.0, gt_img / 255.0, mask_arr=mask_arr))

    loss_arr = np.array(loss_lst)
    auc_arr = np.array(auc_lst)
    acc_arr = np.array(acc_lst)
    sen_arr = np.array(sen_lst)
    fdr_arr = np.array(fdr_lst)
    spe_arr = np.array(spe_lst)
    iou_arr = np.array(iou_lst)
    dice_arr = np.array(dice_lst)

    print("Loss - mean: " + str(loss_arr.mean()) + "\tstd: " + str(loss_arr.std()))
    print("AUC - mean: " + str(auc_arr.mean()) + "\tstd: " + str(auc_arr.std()))
    print("ACC - mean: " + str(acc_arr.mean()) + "\tstd: " + str(acc_arr.std()))
    print("SEN - mean: " + str(sen_arr.mean()) + "\tstd: " + str(sen_arr.std()))
    print("FDR - mean: " + str(fdr_arr.mean()) + "\tstd: " + str(fdr_arr.std()))
    print("SPE - mean: " + str(spe_arr.mean()) + "\tstd: " + str(spe_arr.std()))
    print("IOU - mean: " + str(iou_arr.mean()) + "\tstd: " + str(iou_arr.std()))
    print("Dice - mean: " + str(dice_arr.mean()) + "\tstd: " + str(dice_arr.std()))

    return loss_arr, auc_arr, acc_arr, sen_arr, fdr_arr, spe_arr, iou_arr, dice_arr

if __name__ == '__main__':
    g = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 1])
    s = np.array([0, 1, 0, 1, 1, 0, 0, 0, 0, 1])
    print(calc_acc(g,s))
    print(calc_iou(g,s))
    print(calc_dice(g,s))
    auc_score1 = metrics.roc_auc_score(g, s)
    auc_score2 = metrics.roc_auc_score(s, g)
    print(auc_score1,auc_score2) # 0.75
