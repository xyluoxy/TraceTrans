import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math



class NCC:
    """Local (windowed) normalized cross-correlation loss.

    For every voxel the squared correlation between ``y_true`` and ``y_pred`` is computed
    over a local window (via convolution with a box filter) and averaged. The value is
    negated so that minimizing it maximizes correlation.

    Parameters:
        win (list[int] | None): window size per spatial dimension. ``None`` uses a window
            of ``9`` along every dimension.

    ``loss(y_true, y_pred)`` expects tensors shaped ``[B, C, *vol_shape]`` with 1-3
    spatial dimensions and returns a scalar in ``[-1, 0]`` (lower is better).

    Note:
        The box filter is created with a hard-coded ``.to("cuda")``, so this loss
        currently requires the inputs to be on a CUDA device.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class MSE:
    """Mean squared error loss.

    ``loss(y_true, y_pred)`` returns the mean of ``(y_true - y_pred) ** 2`` over all
    elements as a scalar tensor.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """N-D soft Dice loss for segmentation.

    ``loss(y_true, y_pred)`` computes the soft Dice overlap between two probability maps
    shaped ``[B, C, *vol_shape]`` (summed over the spatial axes, averaged over batch and
    channels) and returns its negative, so lower is better.
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """N-D spatial-gradient (smoothness) regularizer.

    Penalizes the finite-difference gradients of a field along every spatial dimension,
    typically applied to a deformation/flow field to encourage smooth, locally coherent
    warps.

    Parameters:
        penalty (str): ``'l1'`` (absolute differences) or ``'l2'`` (squared differences).
        loss_mult (float | None): optional multiplier applied to the final value, e.g. to
            compensate for the resolution at which the flow is predicted.

    ``loss(_, y_pred)`` ignores its first argument and returns the mean gradient penalty
    over the batch as a scalar.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()
    


class SoftMutualInformation:
    """Differentiable (soft) normalized mutual information loss.

    The joint histogram of ``y_true`` and ``y_pred`` is approximated with a differentiable
    Parzen-window estimate: intensities (assumed to lie in ``[-1, 1]``) are soft-assigned
    to ``num_bins`` bin centers through a Gaussian kernel, and the joint / marginal
    entropies give ``NMI = (H_f + H_m) / H_fm``. The mean NMI over the batch is negated so
    that minimizing the loss maximizes shared information -- useful as a multi-modal /
    cross-domain image similarity term.

    Parameters:
        num_bins (int): number of intensity bins for the histogram. Default ``32``.
        sigma (float | None): standard deviation of the Gaussian kernel. ``None`` picks a
            value tied to the bin width (``(2 / num_bins) / sqrt(2 ln 2)``).
    """
    def __init__(self, num_bins=32, sigma=None):
        """
        initializer.
        Parameter:
            num_bins: number of bins.
            sigma: standard deviation of the Gaussian (Parzen) kernel.
        """
        self.num_bins = num_bins
        if sigma == None:
            # self.sigma = (1.0 / num_bins) / np.sqrt(2 * np.log(2))
            self.sigma = (2.0 / num_bins) / np.sqrt(2 * np.log(2))
        else:
            self.sigma = sigma

    def gaussian(self, y):
        const = torch.sqrt(torch.tensor(2 * math.pi, device=y.device, dtype=y.dtype))
        coeff = 1 / (self.sigma * const)
        exponent = - (y ** 2) / (2 * self.sigma ** 2)
        return coeff * torch.exp(exponent)

    def loss(self, y_true, y_pred):
        y_true_gaussians = []
        y_pred_gaussians = []
        for i in range(self.num_bins):
            # center_i = (i + 0.5) / self.num_bins
            center_i = -1.0 + (i + 0.5) * 2.0 / self.num_bins
            y_true_gaussians.append(self.gaussian(y_true - center_i))
            y_pred_gaussians.append(self.gaussian(y_pred - center_i))
        y_true_gaussians = torch.stack(y_true_gaussians, dim=0)  # [num_bins, B, C, H, W]
        y_pred_gaussians = torch.stack(y_pred_gaussians, dim=0)
        h = torch.einsum('nbchw,mbchw->bnm', y_true_gaussians, y_pred_gaussians) # use einsum to calculate the histogram
        
        p = h / torch.sum(h, dim=[1, 2])[:, None, None]
        p_f = torch.sum(p, dim=2)
        p_m = torch.sum(p, dim=1)

        eps = 1e-8
        H_fm = torch.sum(- p * torch.log(p + eps), dim=[1, 2])
        H_f = torch.sum(- p_f * torch.log(p_f + eps), dim=1)
        H_m = torch.sum(- p_m * torch.log(p_m + eps), dim=1)
        return - torch.mean((H_f + H_m) / H_fm)
    

# From CycleGAN
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real).to(prediction.device)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
