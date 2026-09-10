from typing import Tuple, Union, Optional
from typing import List, Union, Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as nnf

import neurite as ne


class SpatialTransformer(nn.Module):
    """Warp an image with a dense displacement (deformation) field.

    A fixed identity sampling grid is created once for the given spatial ``size`` and
    cached as a (non-persistent) buffer. On every ``forward`` the displacement field is
    added to that grid, the result is normalized to the ``[-1, 1]`` range expected by
    :func:`torch.nn.functional.grid_sample`, and the moving image is resampled at the
    displaced coordinates.

    Parameters:
        size (tuple[int]): spatial shape of the images to be warped, e.g. ``(H, W)``.
        interpolation_mode (str): ``grid_sample`` interpolation, ``"bilinear"`` or
            ``"nearest"``. Default ``"bilinear"``.
        align_corners (bool): passed straight through to ``grid_sample``. Default ``False``.
        device (str | torch.device): device on which the identity grid is allocated.

    Forward:
        moving_image (Tensor): image to warp, shape ``[B, C, *size]``.
        deformation_field (Tensor): per-voxel displacements, shape ``[B, ndim, *size]``
            (channel-first; internally moved to channel-last).

    Returns:
        Tensor: the warped image, same shape as ``moving_image``. ``padding_mode="border"``
        is used for out-of-range samples.

    Note:
        The ``grid[..., [1, 0]]`` axis swap in ``forward`` is a quick fix to match the
        (x, y) ordering ``grid_sample`` expects; keep it in sync with how the deformation
        field is produced upstream.
    """

    def __init__(
        self,
        size: Tuple[int],
        interpolation_mode: str = "bilinear",
        align_corners: bool = False,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()

        self.size = size
        self.device = device
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners

        # Make identity grid (the grid to later warp with deformation field) and register as a
        # buffer (without saving to `state_dict`: persistent=False)
        self.register_buffer(
            name='identity_grid',
            tensor=ne.utils.grid(size=size, device=device),
            persistent=False
        )

    def forward(
        self,
        moving_image: torch.Tensor,
        deformation_field: torch.Tensor
    ) -> torch.Tensor:
        # Validate the dimensions of the input
        if moving_image.dim() < 4 or deformation_field.dim() != moving_image.dim():
            raise ValueError(
                f"Expected `moving_image` to have at least 4 dimensions and for `flow field` to "
                f"match `moving_image` dimensions, got moving_image.dim()={moving_image.dim()}, "
                f"deformation_field.dim()={deformation_field.dim()}"
            )
        
        deformation_field = deformation_field.moveaxis(1, -1).contiguous()

        # Warp the identity grid with the deformation field
        warped_grid = self.identity_grid + deformation_field

        # Normalize the axes so the range does not exceed the interval [-1, 1]
        warped_grid = self._normalize_warped_grid(warped_grid)

        # Sample grid
        warped_image = nnf.grid_sample(
            input=moving_image,
            grid=warped_grid[:, :, :, [1,0]], # [QUICK FIX]
            mode=self.interpolation_mode,
            align_corners=self.align_corners,
            padding_mode="border"
        )

        return warped_image

    def _normalize_warped_grid(
        self,
        warped_grid: torch.Tensor
    ) -> torch.Tensor:

        for i, dim in enumerate(self.size):

            # Rescale each dimension individually
            warped_grid[..., i] = 2 * (warped_grid[..., i] / (dim - 1) - 0.5)

        return warped_grid


class IntegrateVelocityField(nn.Module):
    """Integrate a stationary velocity field into a displacement field by scaling and squaring.

    A stationary velocity field ``v`` is turned into a (near-)diffeomorphic displacement
    field ``phi`` by first scaling it down by ``1 / 2**steps`` and then repeatedly
    composing it with itself ``steps`` times via ``v <- v + warp(v, v)``. With
    ``steps == 0`` the field is returned unchanged, so this module is a no-op unless
    ``integration_steps`` is set on the parent network.

    Parameters:
        shape (tuple): spatial shape of the field, e.g. ``(H, W)``; used to build the
            internal :class:`SpatialTransformer`.
        steps (int): number of squaring steps, must be ``>= 0``. Default ``1``.
        interpolation_mode (str): interpolation used when composing the field.
        align_corners (bool): forwarded to the internal spatial transformer.
        device (str): device for the internal spatial transformer.

    Forward:
        velocity_field (Tensor): stationary velocity field, shape ``[B, ndim, *shape]``.

    Returns:
        Tensor: the integrated displacement field, same shape as the input.
    """

    def __init__(
        self, shape: tuple,
        steps: int = 1,
        interpolation_mode: str = "bilinear",
        align_corners: bool = False,
        device: str = "cpu"
    ):
        super().__init__()

        if steps < 0:
            raise ValueError(f"steps should be >= 0, found: {steps}")

        self.steps = steps
        self.scale = 1.0 / (2 ** self.steps)  # Initial downscaling factor

        # Make the transformer which will perform the warping operation
        self.transformer = SpatialTransformer(shape, interpolation_mode, align_corners, device)

    def forward(self, velocity_field: torch.Tensor) -> torch.Tensor:
        # Apply initial scaling to the velocity field
        velocity_field = velocity_field * self.scale

        # Integration loop
        for _ in range(self.steps):

            # Recursive integration step
            velocity_field = velocity_field + self.transformer(velocity_field, velocity_field)

        return velocity_field


class ResizeDisplacementField(nn.Module):
    """Resample a displacement field to a new resolution and rescale its magnitudes.

    The field is spatially interpolated by ``scale_factor`` and its vector values are
    multiplied by the same factor, so the displacements stay consistent with the new
    grid spacing. This is used to move a low-resolution flow (e.g. one predicted at the
    bottleneck) back to full image resolution.

    Parameters:
        scale_factor (float | int | ne.samplers.Sampler): resize factor; a plain number
            is wrapped with ``ne.samplers.Fixed`` and sampled once per forward pass, so a
            stochastic sampler may also be passed. Default ``1.0`` (identity).
        interpolation_mode (str): mode for :func:`torch.nn.functional.interpolate`.
        align_corners (bool): forwarded to ``interpolate``. Default ``True``.

    Forward:
        displacement_field (Tensor): shape ``[B, ndim, H, W]``.

    Returns:
        Tensor: the resized and magnitude-scaled displacement field.
    """

    def __init__(
        self,
        scale_factor: Optional[Union[float, int, ne.samplers.Sampler]] = 1.0,
        interpolation_mode: str = "bilinear",
        align_corners: bool = True,
    ):
        super().__init__()
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners
        self.scale_factor = ne.samplers.Fixed.make(scale_factor)

    def forward(self, displacement_field: torch.Tensor) -> torch.Tensor:
        # Sample from the scaling sampler.
        scale_factor = self.scale_factor()

        resized_displacement_field = nnf.interpolate(
            displacement_field * scale_factor,  # Scale the magnitudes of the displacement field
            scale_factor=scale_factor,
            mode=self.interpolation_mode,
            align_corners=self.align_corners,
        )

        return resized_displacement_field
    

class TwoDecoderUNet(nn.Module):
    """U-Net with a shared encoder and two task-specific decoders.

    A single downsampling path (with optional residual/skip connections) and a bottleneck
    block are shared by two independent upsampling paths:

    * ``mode='flow'`` -- runs the ``upsampling_conv_blocks`` and ``out_layer``
      (``out_channels`` outputs, ``final_activation``, e.g. Sigmoid). Its output feeds the
      flow/deformation head of the network.
    * ``mode='gen'`` -- runs a second, separately initialized set of upsampling blocks
      (``gen_upsampling_blocks``) and ``gen_out_layer`` (``in_channels`` outputs, ``Tanh``)
      to synthesize a translated image.

    Both decoders consume the same encoder skip connections, so the encoder is trained
    jointly by the image-translation and registration objectives.

    Parameters:
        ndim (int): spatial dimensionality (2 for images).
        in_channels (int): number of input channels; also the ``gen`` decoder output width.
        out_channels (int): number of ``flow`` decoder output channels.
        padding_mode: convolution padding mode, ``'zeros' | 'replicate' | 'reflect'``.
        upsample_mode: upsampling strategy, ``'linear' | 'transposed' | 'nearest'``.
        nb_features (list[int]): channel count per encoder/decoder level (defines depth).
        normalizations: normalization layer(s), a single spec broadcast to every level or
            a per-level list.
        activations: activation layer(s), broadcast or per-level list. Default ``nn.ReLU``.
        order (str): conv-block layer ordering string passed to ``neurite`` (e.g. ``'caca'``).
        final_activation: activation on the ``flow`` output layer. Default ``nn.Sigmoid()``.
        residual_connections (bool): whether to use encoder->decoder skip connections.

    Forward:
        feature_tensor (Tensor): input image/feature map, shape ``[B, in_channels, *spatial]``.
        return_bottleneck (bool): if ``True`` return the bottleneck features and skip both
            decoders.
        mode (str): ``'flow'`` or ``'gen'`` (see above); anything else raises ``ValueError``.

    Returns:
        Tensor: bottleneck features (``return_bottleneck``), the ``flow`` head output, or
        the ``gen`` head output, depending on the arguments.
    """

    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        padding_mode: Literal['zeros', 'replicate', 'reflect'] = 'zeros',
        upsample_mode: Literal['linear', 'transposed', 'nearest'] = 'linear',
        nb_features: List[int] = (16, 16, 16, 16, 16),
        normalizations: Union[List[Union[Callable, str]], Callable, str, None] = None,
        activations: Union[List[Union[Callable, str]], Callable, str, None] = nn.ReLU,
        order: str = 'caca',
        final_activation: Union[str, nn.Module, None] = nn.Sigmoid(),
        residual_connections: bool = True,
    ):
        super().__init__()

        # Storing some attributes that might be useful later on
        self.ndim = ndim
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Make `residual_connections` an attribute as we will need it later in forward pass
        self.residual_connections = residual_connections
        
        # Store order for later use
        self.order = order

        # Normalization layers
        if not isinstance(normalizations, list):
            self.normalizations = [normalizations] * len(nb_features)

        # Activation layers
        if not isinstance(activations, list):
            self.activations = [activations] * len(nb_features)

        # Original sequence for downsampling conv blocks
        self.nb_features = [in_channels, *nb_features]

        # Inverted sequence for upsampling conv blocks
        self.reversed_features = list(reversed(nb_features))

        # Downsampling convolutional blocks
        self.downsampling_conv_blocks = ne.utils.make_downsampling_conv_blocks(
            ndim=ndim,
            nb_features=self.nb_features,
            normalizations=self.normalizations,
            activations=self.activations,
            order=order,
            return_residual=residual_connections,
            padding_mode=padding_mode,
        )

        # Convolutional block between downsampling and upsampling arms (lowest resolution)
        self.lowest_resolution_conv_block = ne.modules.ConvBlock(
            ndim=ndim,
            in_channels=self.nb_features[-1],
            out_channels=self.nb_features[-1],
            order=order,
            padding_mode=padding_mode,
        )

        # Upsampling convolutional blocks
        self.upsampling_conv_blocks = ne.utils.make_upsampling_conv_blocks(
            ndim=ndim,
            nb_features=self.reversed_features,
            normalizations=self.normalizations,
            activations=self.activations,
            order=order,
            upsample_kernel_size=2,
            upsample_stride=2,
            upsample_padding=0,
            accepts_residuals=residual_connections,
            padding_mode=padding_mode,
            upsample_mode=upsample_mode
        )

        # Final convolutional block
        self.out_layer = ne.modules.ConvBlock(
            ndim=ndim,
            in_channels=nb_features[0],
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            activation=final_activation,
            padding_mode=padding_mode,
        )
        
        # Initialize generator decoder components
        self._init_gen_decoder_components()

    def forward(self, feature_tensor: torch.Tensor, return_bottleneck: bool = False, mode: str = 'flow'):
        # Downsampling path
        skip_connections = []

        for downsampling_conv_block in self.downsampling_conv_blocks:
            if self.residual_connections:
                feature_tensor, residual = downsampling_conv_block(feature_tensor)
                skip_connections.append(residual)  # Save for skip connection
            else:
                feature_tensor = downsampling_conv_block(feature_tensor)

        # Convolutional block between downsampling and upsampling arms (lowest resolution)
        bottleneck_features = self.lowest_resolution_conv_block(feature_tensor)

        # If only bottleneck features are requested, return them
        if return_bottleneck:
            return bottleneck_features

        # Upsampling path based on mode
        if mode == 'flow':
            # Original flow mode
            feature_tensor = bottleneck_features
            for i, upsampling_conv_block in enumerate(self.upsampling_conv_blocks):
                if self.residual_connections:
                    skip = skip_connections[-(i + 1)]
                    feature_tensor = upsampling_conv_block(feature_tensor, skip)
                else:
                    feature_tensor = upsampling_conv_block(feature_tensor)

            # Output layer
            feature_tensor = self.out_layer(feature_tensor)
            return feature_tensor
            
        elif mode == 'gen':
            # Generation mode with separate decoder
            return self._gen_decoder_forward(bottleneck_features, skip_connections)
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'flow' or 'gen'.")

    def _init_gen_decoder_components(self):
        """Initialize the generator decoder components during model initialization."""
        # Create new upsampling blocks for generator (same structure as flow but separate)
        self.gen_upsampling_blocks = ne.utils.make_upsampling_conv_blocks(
            ndim=self.ndim,
            nb_features=self.reversed_features,
            normalizations=self.normalizations,
            activations=self.activations,
            order=self.order,
            upsample_kernel_size=2,
            upsample_stride=2,
            upsample_padding=0,
            accepts_residuals=self.residual_connections,
            padding_mode='zeros',
            upsample_mode='linear'
        )
        
        # Different output layer for generator (Tanh activation for image generation)
        self.gen_out_layer = ne.modules.ConvBlock(
            ndim=self.ndim,
            in_channels=self.reversed_features[-1],  # Use last decoder feature count
            out_channels=self.in_channels,  # Output same channels as input
            kernel_size=1,
            padding=0,
            activation=torch.nn.Tanh(),  # Use Tanh for image generation
            padding_mode='zeros',
        )
    
    def _gen_decoder_forward(self, bottleneck_features, skip_connections):
        """Forward pass through the generator decoder with skip connections."""
        # Use the same logic as flow mode, just with different output layer
        feature_tensor = bottleneck_features
        for i, upsampling_conv_block in enumerate(self.gen_upsampling_blocks):
            if self.residual_connections:
                skip = skip_connections[-(i + 1)]
                feature_tensor = upsampling_conv_block(feature_tensor, skip)
            else:
                feature_tensor = upsampling_conv_block(feature_tensor)

        # Output layer
        feature_tensor = self.gen_out_layer(feature_tensor)
        return feature_tensor

