import torch
import random

def randn_sampling(maxint, sample_rate):
    B, C, D, H, W = maxint
    sample_size = int(D*H*W*sample_rate)
    ramdom_index = torch.LongTensor([[],[],[],[],[]])
    # ramdom_index = torch.tensor(ramdom_index, dtype=torch.int)
    for i in range(B):
        b = torch.ones((sample_size,1)).int() - 1 + i
        d = torch.randint(D, size=(sample_size, 1))
        h = torch.randint(H, size=(sample_size, 1))
        w = torch.randint(W, size=(sample_size, 1))
        for j in range(C):
            c = torch.ones((sample_size,1)).int() - 1 + j
            tmpindex = torch.cat([b.long(), c.long(), d.long(), h.long(), w.long()], dim=1).t()
            # tmpindex = torch.tensor(tmpindex, dtype=torch.long)
            ramdom_index = torch.cat([ramdom_index,tmpindex], dim=1)
    return ramdom_index.long()


def random_mask(soft_target, pre, sample_rate):
    B, C, D, H, W = soft_target.size()
    ramdom_index = randn_sampling((B, C, D, H, W), sample_rate)
    ramdom_index_t = torch.split(ramdom_index, 1, dim=0)
    soft_target_mask = soft_target.index_put(ramdom_index_t, values=torch.tensor(0.))
    pre_mask = pre.index_put(ramdom_index_t, values=torch.tensor(0.))

    return soft_target_mask, pre_mask


def random_mask_all(soft_target, flair_pred, t1ce_pred, t1_pred, t2_pred, sample_rate):
    B, C, D, H, W = soft_target.size()
    ramdom_index = randn_sampling((B, C, D, H, W), sample_rate)
    ramdom_index_t = torch.split(ramdom_index, 1, dim=0)
    soft_target_mask = soft_target.index_put(ramdom_index_t, values=torch.tensor(0.))
    flair_pred_mask = flair_pred.index_put(ramdom_index_t, values=torch.tensor(0.))
    t1ce_pred_mask = t1ce_pred.index_put(ramdom_index_t, values=torch.tensor(0.))
    t1_pred_mask = t1_pred.index_put(ramdom_index_t, values=torch.tensor(0.))
    t2_pred_mask = t2_pred.index_put(ramdom_index_t, values=torch.tensor(0.))

    return soft_target_mask, flair_pred_mask, t1ce_pred_mask,t1_pred_mask,t2_pred_mask

def replace_random_positions_with_zero(map_A, map_B, replace_percentage=0.2):
    assert map_A.size() == map_B.size(), "Input maps must have the same size"
    
    # Flatten the maps
    flattened_A = map_A.contiguous().view(map_A.size(0), -1)
    flattened_B = map_B.contiguous().view(map_B.size(0), -1)
    
    # Calculate the number of positions to replace
    num_elements = flattened_A.size(1)
    num_positions = int(num_elements * replace_percentage)
    
    # Generate random indices to replace
    indices = random.sample(range(num_elements), num_positions)
    
    # Replace positions with zero
    flattened_A[:, indices] = 0
    flattened_B[:, indices] = 0
    
    # Reshape back to the original size
    replaced_A = flattened_A.contiguous().view(map_A.size())
    replaced_B = flattened_B.contiguous().view(map_B.size())
    
    return replaced_A, replaced_B


def softmax_weighted_loss(output, target, num_cls=5):
    target = target.float()
    B, _, H, W, Z = output.size()
    for i in range(num_cls):
        outputi = output[:, i, :, :, :]
        targeti = target[:, i, :, :, :]
        weighted = 1.0 - (torch.sum(targeti, (1,2,3)) * 1.0 / torch.sum(target, (1,2,3,4)))
        weighted = torch.reshape(weighted, (-1,1,1,1)).repeat(1,H,W,Z)
        if i == 0:
            cross_loss = -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
        else:
            cross_loss += -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
    cross_loss = torch.mean(cross_loss)
    return cross_loss