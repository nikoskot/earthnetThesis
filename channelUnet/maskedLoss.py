from torch import nn
import torch.nn.functional as F



class MaskedLoss(nn.Module):
    def __init__(self, distance_type = "L2"):
        super(MaskedLoss, self).__init__()
        self.distance_type = distance_type

    def forward(self, preds,targets,mask):
        assert(preds.shape == targets.shape)
        predsmasked = preds * mask
        targetsmasked = targets * mask

        if self.distance_type == "L2":
            return F.mse_loss(predsmasked,targetsmasked,reduction = 'sum')/ ((mask > 0).sum() + 1)
        elif self.distance_type == "L1":
            return F.l1_loss(predsmasked,targetsmasked,reduction = 'sum')/ ((mask > 0).sum() + 1)
        
class BaseLoss(nn.Module):
    def __init__(self, setting: dict, device):
        super().__init__()

        self.distance = MaskedLoss(distance_type="L1")
        self.device = device

    def forward(self, preds, batch, aux, current_step = None):
        
        logs = {}

        targs = batch["dynamic"][0][:,-preds.shape[1]:,...].to(self.device)
        masks = batch["dynamic_mask"][0][:,-preds.shape[1]:,...].to(self.device)

        dist = self.distance(preds, targs, masks)
        
        logs["distance"] = dist

        loss = dist

        logs["loss"] = loss

        return loss, logs