import torch
import numpy as np



def singel_teacher_uncertainty_np_neg(pre1, pre2, pre3, pre4, current_step, max_step, pre1_logits, pre2_logits, pre3_logits, pre4_logits, pre1_pos, pre2_pos, pre3_pos, pre4_pos, gt):

    pre1, pre2, pre3, pre4 = pre1.cpu(), pre2.cpu(), pre3.cpu(), pre4.cpu()

    pre1_logits, pre2_logits, pre3_logits, pre4_logits = pre1_logits.cpu(), pre2_logits.cpu(), pre3_logits.cpu(), pre4_logits.cpu()

    pre1_pos, pre2_pos, pre3_pos, pre4_pos =  pre1_pos.cpu(), pre2_pos.cpu(), pre3_pos.cpu(), pre4_pos.cpu()

    gt = gt.cpu()


    T1 = 0.75
    T2 = 0.85


    pre1_mask_temple = pre1 > T1

    pre1_mask = np.logical_and(pre1_mask_temple, pre1_pos)



    pre_logits = pre1_logits.clone()

    pre_logits[np.logical_not(pre1_mask)] = 0



    pre2_temp = pre2.clone()
    pre2_temp[pre1_mask] = 0
    pre2_logits_temp = pre2_logits.clone()
    pre2_logits_temp[np.logical_not(pre1_mask)] = 0

    pre3_temp = pre3.clone()
    pre3_temp[pre1_mask] = 0
    pre3_logits_temp = pre3_logits.clone()
    pre3_logits_temp[np.logical_not(pre1_mask)] = 0

    pre4_temp = pre4.clone()
    pre4_temp[pre1_mask] = 0
    pre4_logits_temp = pre4_logits.clone()
    pre4_logits_temp[np.logical_not(pre1_mask)] = 0


    pre2_mask_temple = pre2_temp > T2
    pre3_mask_temple = pre3_temp > T2
    pre4_mask_temple = pre4_temp > T2


    pre2_mask = np.logical_and(pre2_mask_temple, pre2_pos)
    pre3_mask = np.logical_and(pre3_mask_temple, pre3_pos)
    pre4_mask = np.logical_and(pre4_mask_temple, pre4_pos)



    low_weight_mask = np.logical_and(pre4_mask, np.logical_and(pre2_mask, pre3_mask))

    chose_mask = np.logical_not(low_weight_mask)
    pre2_temp[chose_mask] = 0
    pre3_temp[chose_mask] = 0
    pre4_temp[chose_mask] = 0

    pre2_logits_temp[chose_mask] = 0
    pre3_logits_temp[chose_mask] = 0
    pre4_logits_temp[chose_mask] = 0



    pre2_3_4_logits = (pre2_logits_temp + pre3_logits_temp + pre4_logits_temp)/3



    pre1_logits[pre1_mask] = 0
    pre2_logits[pre1_mask] = 0
    pre3_logits[pre1_mask] = 0
    pre4_logits[pre1_mask] = 0

    pre1_logits[low_weight_mask] = 0
    pre2_logits[low_weight_mask] = 0
    pre3_logits[low_weight_mask] = 0
    pre4_logits[low_weight_mask] = 0

    gt[pre1_mask] = 0
    gt[low_weight_mask] = 0




    pre_min_logits = np.min(np.concatenate([pre1_logits, pre2_logits, pre3_logits, pre4_logits],axis=1), axis=1, keepdims=True)


    pre_logits = torch.tensor(pre_logits) + torch.tensor(pre2_3_4_logits) + torch.tensor(pre_min_logits)

    return pre_logits.cuda()



def double_teacher_uncertainty_np_neg(pre1, pre2, pre3, pre4, current_step, max_step, pre1_logits, pre2_logits, pre3_logits, pre4_logits, pre1_pos, pre2_pos, pre3_pos, pre4_pos, gt):

    pre1, pre2, pre3, pre4 = pre1.cpu(), pre2.cpu(), pre3.cpu(), pre4.cpu()

    pre1_logits, pre2_logits, pre3_logits, pre4_logits = pre1_logits.cpu(), pre2_logits.cpu(), pre3_logits.cpu(), pre4_logits.cpu()

    pre1_pos, pre2_pos, pre3_pos, pre4_pos =  pre1_pos.cpu(), pre2_pos.cpu(), pre3_pos.cpu(), pre4_pos.cpu()

    gt = gt.cpu()


    T1 = 0.75
    T2 = 0.85

    pre1_mask_temple = pre1 > T1
    pre2_mask_temple = pre2 > T1

    pre1_mask = np.logical_and(pre1_mask_temple, pre1_pos)
    pre2_mask = np.logical_and(pre2_mask_temple, pre2_pos)

    pre_domain_mask = np.logical_or(pre1_mask, pre2_mask)


    pre_domain_mask1 = np.logical_and(pre1_mask, pre2_mask)

    pre1_temp1 = pre1.clone()
    pre2_temp1 = pre2.clone()

    pre1_logits_temp1 = pre1_logits.clone()
    pre2_logits_temp1 = pre2_logits.clone()

    pre1_temp1[np.logical_not(pre_domain_mask1)] = 0

    pre2_temp1[np.logical_not(pre_domain_mask1)] = 0

    pre1_logits_temp1[np.logical_not(pre_domain_mask1)] = 0
    pre2_logits_temp1[np.logical_not(pre_domain_mask1)] = 0



    max1_21 = np.max(np.concatenate([pre1_logits_temp1, pre2_logits_temp1],axis=1), axis=1, keepdims=True)

    pre_domain1 =  max1_21

    pre_domain_mask2 = np.logical_xor(pre_domain_mask, pre_domain_mask1)

    pre1_temp2 = pre1.clone()
    pre2_temp2 = pre2.clone()

    pre1_logits_temp2 = pre1_logits.clone()
    pre2_logits_temp2 = pre2_logits.clone()

    pre1_temp2[np.logical_not(pre_domain_mask2)] = 0

    pre2_temp2[np.logical_not(pre_domain_mask2)] = 0

    pre1_logits_temp2[np.logical_not(pre_domain_mask2)] = 0
    pre2_logits_temp2[np.logical_not(pre_domain_mask2)] = 0




    max1_22 = np.max(np.concatenate([pre1_logits_temp2, pre2_logits_temp2],axis=1), axis=1, keepdims=True)
    min1_22 = np.min(np.concatenate([pre1_logits_temp2, pre2_logits_temp2],axis=1), axis=1, keepdims=True)
    pre_domain2 =  max1_22 - 0.2*(max1_22 - min1_22)







    pre3_temp = pre3.clone()
    pre3_temp[pre_domain_mask] = 0
    pre3_logits_temp = pre3_logits.clone()
    pre3_logits_temp[pre_domain_mask] = 0
    

    pre4_temp = pre4.clone()
    pre4_temp[pre_domain_mask] = 0
    pre4_logits_temp = pre4_logits.clone()
    pre4_logits_temp[pre_domain_mask] = 0

    pre3_mask_temple = pre3_temp > T2
    pre4_mask_temple = pre4_temp > T2

    pre3_mask = np.logical_and(pre3_mask_temple, pre3_pos)
    pre4_mask = np.logical_and(pre4_mask_temple, pre3_pos)
    



    low_weight_mask = np.logical_and(pre4_mask, pre3_mask)



    chose_mask = np.logical_not(low_weight_mask)
    pre3_temp[chose_mask] = 0
    pre4_temp[chose_mask] = 0

    pre3_logits_temp[chose_mask] = 0
    pre4_logits_temp[chose_mask] = 0





    pre3_4 = (pre3_logits_temp + pre4_logits_temp)/2



    pre1_logits[pre_domain_mask] = 0
    pre2_logits[pre_domain_mask] = 0
    pre3_logits[pre_domain_mask] = 0
    pre4_logits[pre_domain_mask] = 0

    pre1_logits[low_weight_mask] = 0
    pre2_logits[low_weight_mask] = 0
    pre3_logits[low_weight_mask] = 0
    pre4_logits[low_weight_mask] = 0

    gt[pre_domain_mask] = 0

    gt[low_weight_mask] = 0




    pre_min = np.min(np.concatenate([pre1_logits, pre2_logits, pre3_logits, pre4_logits],axis=1), axis=1, keepdims=True)



    



    pre = torch.tensor(pre_domain1) + torch.tensor(pre_domain2) + torch.tensor(pre3_4) + torch.tensor(pre_min)  

    return pre.cuda()
