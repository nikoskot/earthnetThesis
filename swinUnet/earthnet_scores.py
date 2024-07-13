import numpy as np
from scipy.stats import hmean
import torch
from torchmetrics import Metric
from einops import rearrange
from earthnet.parallel_score import CubeCalculator as EN_CubeCalculator
from typing import Any
import torch
from torchmetrics import Metric
from torchmetrics.utilities.exceptions import TorchMetricsUserError
import time


class MetricsUpdateWithoutCompute(Metric):
    r"""
    Delete `batch_val = self.compute()` in `forward()` to reduce unnecessary computation
    """

    def __init__(self, **kwargs: Any):
        super(MetricsUpdateWithoutCompute, self).__init__(**kwargs)

    @torch.jit.unused
    def forward(self, *args: Any, **kwargs: Any):
        """``forward`` serves the dual purpose of both computing the metric on the current batch of inputs but also
        add the batch statistics to the overall accumululating metric state.

        Input arguments are the exact same as corresponding ``update`` method. The returned output is the exact same as
        the output of ``compute``.
        """
        # check if states are already synced
        if self._is_synced:
            raise TorchMetricsUserError(
                "The Metric shouldn't be synced when performing ``forward``. "
                "HINT: Did you forget to call ``unsync`` ?."
            )

        if self.full_state_update or self.full_state_update is None or self.dist_sync_on_step:
            self._forward_full_state_update_without_compute(*args, **kwargs)
        else:
            self._forward_reduce_state_update_without_compute(*args, **kwargs)

    def _forward_full_state_update_without_compute(self, *args: Any, **kwargs: Any):
        """forward computation using two calls to `update` to calculate the metric value on the current batch and
        accumulate global state.

        Doing this secures that metrics that need access to the full metric state during `update` works as expected.
        """
        # global accumulation
        self.update(*args, **kwargs)
        _update_count = self._update_count

        self._to_sync = self.dist_sync_on_step  # type: ignore
        # skip restore cache operation from compute as cache is stored below.
        self._should_unsync = False
        # skip computing on cpu for the batch
        _temp_compute_on_cpu = self.compute_on_cpu
        self.compute_on_cpu = False

        # save context before switch
        cache = {attr: getattr(self, attr) for attr in self._defaults}

        # call reset, update, compute, on single batch
        self._enable_grad = True  # allow grads for batch computation
        self.reset()
        self.update(*args, **kwargs)

        # restore context
        for attr, val in cache.items():
            setattr(self, attr, val)
        self._update_count = _update_count

        # restore context
        self._is_synced = False
        self._should_unsync = True
        self._to_sync = self.sync_on_compute
        self._computed = None
        self._enable_grad = False
        self.compute_on_cpu = _temp_compute_on_cpu

    def _forward_reduce_state_update_without_compute(self, *args: Any, **kwargs: Any):
        """forward computation using single call to `update` to calculate the metric value on the current batch and
        accumulate global state.

        This can be done when the global metric state is a sinple reduction of batch states.
        """
        # store global state and reset to default
        global_state = {attr: getattr(self, attr) for attr in self._defaults.keys()}
        _update_count = self._update_count
        self.reset()

        # local synchronization settings
        self._to_sync = self.dist_sync_on_step
        self._should_unsync = False
        _temp_compute_on_cpu = self.compute_on_cpu
        self.compute_on_cpu = False
        self._enable_grad = True  # allow grads for batch computation

        # calculate batch state and compute batch value
        self.update(*args, **kwargs)

        # reduce batch and global state
        self._update_count = _update_count + 1
        with torch.no_grad():
            self._reduce_states(global_state)

        # restore context
        self._is_synced = False
        self._should_unsync = True
        self._to_sync = self.sync_on_compute
        self._computed = None
        self._enable_grad = False
        self.compute_on_cpu = _temp_compute_on_cpu


class EarthNet2021Score(Metric):

    default_layout = "NHWCT"
    default_channel_axis = 3
    channels = 4

    def __init__(self,
                 layout: str = "NTHWC",
                 eps: float = 1e-4,
                 dist_sync_on_step: bool = False, ):
        super(EarthNet2021Score, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self.layout = layout
        self.eps = eps

        self.add_state("MAD",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("OLS",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("EMD",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("SSIM",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        # does not count if NaN
        self.add_state("num_MAD",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("num_OLS",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("num_EMD",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("num_SSIM",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

    @property
    def einops_default_layout(self):
        if not hasattr(self, "_einops_default_layout"):
            self._einops_default_layout = " ".join(self.default_layout)
        return self._einops_default_layout

    @property
    def einops_layout(self):
        if not hasattr(self, "_einops_layout"):
            self._einops_layout = " ".join(self.layout)
        return self._einops_layout

    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        r"""

        Parameters
        ----------
        pred, target:   torch.Tensor
            With the first dim as batch dim, and 4 channels (RGB and infrared)
        mask:   torch.Tensor
            With the first dim as batch dim, and 1 channel
        """
        pred_np = rearrange(pred.detach(), f"{self.einops_layout} -> {self.einops_default_layout}").cpu().numpy()
        target_np = rearrange(target.detach(), f"{self.einops_layout} -> {self.einops_default_layout}").cpu().numpy()
        # layout = "NHWCT"
        if mask is None:
            mask_np = np.ones_like(target_np)
        else:
            mask_np = torch.repeat_interleave(
                rearrange(1 - mask.detach(), f"{self.einops_layout} -> {self.einops_default_layout}"),
                repeats=self.channels, dim=self.default_channel_axis).cpu().numpy()
        for preds, targs, masks in zip(pred_np, target_np, mask_np):
            # Code is adapted from `load_file()` in ./earthnet_toolkit/parallel_score.py
            preds[preds < 0] = 0
            preds[preds > 1] = 1

            targs[np.isnan(targs)] = 0
            targs[targs > 1] = 1
            targs[targs < 0] = 0

            ndvi_preds = ((preds[:, :, 3, :] - preds[:, :, 2, :]) / (preds[:, :, 3, :] + preds[:, :, 2, :] + 1e-6))[:,
                         :, np.newaxis, :]
            ndvi_targs = ((targs[:, :, 3, :] - targs[:, :, 2, :]) / (targs[:, :, 3, :] + targs[:, :, 2, :] + 1e-6))[:,
                         :, np.newaxis, :]
            ndvi_masks = masks[:, :, 0, :][:, :, np.newaxis, :]
            # Code is adapted from `get_scores()` in ./earthnet_toolkit/parallel_score.py
            debug_info = {}
            mad, debug_info["MAD"] = EN_CubeCalculator.MAD(preds, targs, masks)
            ols, debug_info["OLS"] = EN_CubeCalculator.OLS(ndvi_preds, ndvi_targs, ndvi_masks)
            emd, debug_info["EMD"] = EN_CubeCalculator.EMD(ndvi_preds, ndvi_targs, ndvi_masks)
            ssim, debug_info["SSIM"] = EN_CubeCalculator.SSIM(preds, targs, masks)
            # does not count if NaN
            if mad is not None and not np.isnan(mad):
                self.MAD += mad
                self.num_MAD += 1
            if ols is not None and not np.isnan(ols):
                self.OLS += ols
                self.num_OLS += 1
            if emd is not None and not np.isnan(emd):
                self.EMD += emd
                self.num_EMD += 1
            if ssim is not None and not np.isnan(ssim):
                self.SSIM += ssim
                self.num_SSIM += 1

    def compute(self):
        MAD_mean = (self.MAD / (self.num_MAD + self.eps)).cpu().item()
        OLS_mean = (self.OLS / (self.num_OLS + self.eps)).cpu().item()
        EMD_mean = (self.EMD / (self.num_EMD + self.eps)).cpu().item()
        SSIM_mean = (self.SSIM / (self.num_SSIM + self.eps)).cpu().item()
        ENS = hmean([MAD_mean, OLS_mean, EMD_mean, SSIM_mean])
        return {
            "MAD": MAD_mean,
            "OLS": OLS_mean,
            "EMD": EMD_mean,
            "SSIM":SSIM_mean,
            "EarthNetScore": ENS,
        }

class EarthNet2021ScoreUpdateWithoutCompute(MetricsUpdateWithoutCompute):

    default_layout = "NHWCT"
    default_channel_axis = 3
    channels = 4

    def __init__(self,
                 layout: str = "NTHWC",
                 eps: float = 1e-4,
                 dist_sync_on_step: bool = False, ):
        super(EarthNet2021ScoreUpdateWithoutCompute, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self.layout = layout
        self.eps = eps

        self.add_state("MAD",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("OLS",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("EMD",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("SSIM",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        # does not count if NaN
        self.add_state("num_MAD",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("num_OLS",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("num_EMD",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("num_SSIM",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

    @property
    def einops_default_layout(self):
        if not hasattr(self, "_einops_default_layout"):
            self._einops_default_layout = " ".join(self.default_layout)
        return self._einops_default_layout

    @property
    def einops_layout(self):
        if not hasattr(self, "_einops_layout"):
            self._einops_layout = " ".join(self.layout)
        return self._einops_layout

    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        r"""

        Parameters
        ----------
        pred, target:   torch.Tensor
            With the first dim as batch dim, and 4 channels (RGB and infrared)
        mask:   torch.Tensor
            With the first dim as batch dim, and 1 channel
        """
        pred_np = rearrange(pred.detach(), f"{self.einops_layout} -> {self.einops_default_layout}").cpu().numpy()
        target_np = rearrange(target.detach(), f"{self.einops_layout} -> {self.einops_default_layout}").cpu().numpy()
        # layout = "NHWCT"
        if mask is None:
            mask_np = np.ones_like(target_np)
        else:
            mask_np = torch.repeat_interleave(
                rearrange(1 - mask.detach(), f"{self.einops_layout} -> {self.einops_default_layout}"),
                repeats=self.channels, dim=self.default_channel_axis).cpu().numpy()
        for preds, targs, masks in zip(pred_np, target_np, mask_np):
            # Code is adapted from `load_file()` in ./earthnet_toolkit/parallel_score.py
            preds[preds < 0] = 0
            preds[preds > 1] = 1

            targs[np.isnan(targs)] = 0
            targs[targs > 1] = 1
            targs[targs < 0] = 0

            ndvi_preds = ((preds[:, :, 3, :] - preds[:, :, 2, :]) / (preds[:, :, 3, :] + preds[:, :, 2, :] + 1e-6))[:,
                         :, np.newaxis, :]
            ndvi_targs = ((targs[:, :, 3, :] - targs[:, :, 2, :]) / (targs[:, :, 3, :] + targs[:, :, 2, :] + 1e-6))[:,
                         :, np.newaxis, :]
            ndvi_masks = masks[:, :, 0, :][:, :, np.newaxis, :]
            # Code is adapted from `get_scores()` in ./earthnet_toolkit/parallel_score.py
            debug_info = {}
            mad, debug_info["MAD"] = EN_CubeCalculator.MAD(preds, targs, masks)
            ols, debug_info["OLS"] = EN_CubeCalculator.OLS(ndvi_preds, ndvi_targs, ndvi_masks)
            emd, debug_info["EMD"] = EN_CubeCalculator.EMD(ndvi_preds, ndvi_targs, ndvi_masks)
            ssim, debug_info["SSIM"] = EN_CubeCalculator.SSIM(preds, targs, masks)
            # does not count if NaN
            if mad is not None and not np.isnan(mad):
                self.MAD += mad
                self.num_MAD += 1
            if ols is not None and not np.isnan(ols):
                self.OLS += ols
                self.num_OLS += 1
            if emd is not None and not np.isnan(emd):
                self.EMD += emd
                self.num_EMD += 1
            if ssim is not None and not np.isnan(ssim):
                self.SSIM += ssim
                self.num_SSIM += 1

    def compute(self):
        MAD_mean = (self.MAD / (self.num_MAD + self.eps)).cpu().item()
        OLS_mean = (self.OLS / (self.num_OLS + self.eps)).cpu().item()
        EMD_mean = (self.EMD / (self.num_EMD + self.eps)).cpu().item()
        SSIM_mean = (self.SSIM / (self.num_SSIM + self.eps)).cpu().item()
        ENS = hmean([MAD_mean, OLS_mean, EMD_mean, SSIM_mean])
        return {
            "MAD": MAD_mean,
            "OLS": OLS_mean,
            "EMD": EMD_mean,
            "SSIM":SSIM_mean,
            "EarthNetScore": ENS,
        }
    
def preprocess(pred, targ):
    """
        From parallel_scores.py load_file()
    
        Load a single target cube and a matching prediction

        Args:
            pred_filepath (Path): Path to predicted cube
            targ_filepath (Path): Path to target cube

        Returns:
            Sequence[np.ndarray]: preds, targs, masks, ndvi_preds, ndvi_targs, ndvi_masks
    """
    # mask = y[:, 4, :, :, :].unsqueeze(1).permute(0, 3, 4, 1, 2)
    # validENSCalculator.update(pred.permute(0, 3, 4, 1, 2)[:, :, :, :4, :], y.permute(0, 3, 4, 1, 2)[:, :, :, :4, :], mask=mask)

    pred = pred.detach().cpu().numpy()
    targ = targ.detach().cpu().numpy()
    
    preds = pred[:,:,:,:4,:]
    print(preds.shape)
    targs = targ[:,:,:,:4,:]
    print(targs.shape)
    masks = ((1 - targ[:,:,:,-1,:])[:,:,:,np.newaxis,:])
    masks = np.repeat(masks, preds.shape[3], axis=3)
    print(masks.shape)


    if preds.shape[-1] < targs.shape[-1]:
        targs = targs[:,:,:,-preds.shape[-1]:]
        masks = masks[:,:,:,-preds.shape[-1]:]
    
    assert(preds.shape == targs.shape)

    preds[preds < 0] = 0
    preds[preds > 1] = 1

    targs[np.isnan(targs)] = 0
    targs[targs > 1] = 1
    targs[targs < 0] = 0

    ndvi_preds = ((preds[:,:,:,3,:] - preds[:,:,:,2,:])/(preds[:,:,:,3,:] + preds[:,:,:,2,:] + 1e-6))[:,:,:,np.newaxis,:]
    ndvi_targs = ((targs[:,:,:,3,:] - targs[:,:,:,2,:])/(targs[:,:,:,3,:] + targs[:,:,:,2,:] + 1e-6))[:,:,:,np.newaxis,:]
    ndvi_masks = masks[:,:,:,0,:][:,:,:,np.newaxis,:]

    return preds, targs, masks, ndvi_preds, ndvi_targs, ndvi_masks

def compute_scores(pred, targ):

    caclulator = EN_CubeCalculator()

    preds, targs, masks, ndvi_preds, ndvi_targs, ndvi_masks = preprocess(pred, targ)
    mads = []
    olss = []
    emds = []
    ssims = []

    for b in range(preds.shape[0]):

        mad, _ = caclulator.MAD(preds[b,...], targs[b,...], masks[b,...])

        ols, _ = caclulator.OLS(ndvi_preds[b,...], ndvi_targs[b,...], ndvi_masks[b,...])

        emd, _ = caclulator.EMD(ndvi_preds[b,...], ndvi_targs[b,...], ndvi_masks[b,...])

        ssim, _ = caclulator.SSIM(preds[b,...], targs[b,...], masks[b,...])

        mads.append(mad)
        olss.append(ols)
        emds.append(emd)
        ssims.append(ssim)
    
    return np.mean(mads), np.mean(olss), np.mean(emds), np.mean(ssims)

if __name__ == "__main__":

    dummy = torch.rand(16, 128, 128, 5, 20).to(torch.device('cuda')) # B, H, W, C, T
    dummy[:, :, :, 4, :] = 0

    start = time.time()
    meanMAD, meanOLSS, meanEMD, meanSSIM = compute_scores(dummy, dummy)
    end = time.time()

    print("Time to calculate scores {}".format(end - start))