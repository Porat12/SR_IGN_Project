"""Parameter-free Back-Projection layer with cached downsampling matrices.

Implements iterative back-projection (IBP) for super-resolution:
    x_out = x_hr + A⁺ · (x_lr − A · x_hr)

where A is a bicubic downsampling operator (Toeplitz matrix) and A⁺ is its
pseudoinverse. This corrects the HR estimate so that downsampling it recovers
the LR input exactly (up to numerical precision).

The Toeplitz matrix is built via the standard Keys cubic kernel (a=-0.5) with
symmetric boundary padding, matching MATLAB's imresize convention. Each row i
of the matrix holds the kernel weights that map HR pixels to output pixel i.

Matrices are built once per spatial size and cached, so repeated calls at the
same resolution have negligible overhead.
"""
import logging
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BackProjectionLayer(nn.Module):
    """Back-Projection layer with cached Toeplitz downsampling matrices.

    Given a low-resolution input ``x_lr`` (from the previous level) and a
    high-resolution estimate ``x_hr`` (from the current level), the layer
    computes::

        residual  = x_lr − downscale(x_hr)      # error in LR space
        correction = upscale(residual)            # map error back to HR space
        output    = x_hr + correction             # corrected HR estimate

    The downscale/upscale pair is a separable bicubic operator expressed as
    row and column Toeplitz matrices ``A_rows``, ``A_cols`` and their
    pseudoinverses ``A_rows⁺``, ``A_cols⁺``.  All four matrices are computed
    once per unique (H, W) and cached for the lifetime of the module.

    The Toeplitz rows are derived from the Keys cubic kernel (a=-0.5) with
    symmetric boundary padding, matching MATLAB's ``imresize`` convention.

    This module has **no learnable parameters**.
    """

    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self._cache: Dict[tuple, Tuple[torch.Tensor, ...]] = {}
        self.factor = scale_factor

    # ------------------------------------------------------------------
    # Matrix construction
    # ------------------------------------------------------------------

    def _build_1d_downsample_matrix(
        self,
        img_dim: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build a 1-D Toeplitz downsampling matrix via the Keys cubic kernel.

        Uses the standard bicubic kernel (a=-0.5) with symmetric boundary
        padding, matching MATLAB's imresize convention. Row i of the result
        holds the kernel weights that map HR pixels to output pixel i:
        ``A @ hr_row = lr_row``.

        Args:
            img_dim: Spatial size of the *high-resolution* dimension.
            device:  Device to place the matrix on.

        Returns:
            Toeplitz matrix of shape ``(img_dim // factor, img_dim)``.
        """
        stride = self.factor
        small_dim = img_dim // stride

        def _bicubic_kernel(x: float, a: float = -0.5) -> float:
            ax = abs(x)
            if ax <= 1:
                return (a + 2) * ax**3 - (a + 3) * ax**2 + 1
            elif ax < 2:
                return a * ax**3 - 5 * a * ax**2 + 8 * a * ax - 4 * a
            return 0.0

        kernel_len = stride * 4
        kernel = torch.zeros(kernel_len)
        for i in range(kernel_len):
            x = (1.0 / stride) * (i - math.floor(kernel_len / 2) + 0.5)
            kernel[i] = _bicubic_kernel(x)
        kernel /= kernel.sum()

        half_k = kernel_len // 2
        mat = torch.zeros(small_dim, img_dim)
        for out_idx, center in enumerate(range(stride // 2, img_dim + stride // 2, stride)):
            for j in range(center - half_k, center + half_k):
                # Symmetric boundary padding (matches MATLAB imresize)
                j_c = j
                if j_c < 0:
                    j_c = -j_c - 1
                if j_c >= img_dim:
                    j_c = (img_dim - 1) - (j_c - img_dim)
                mat[out_idx, j_c] += kernel[j - center + half_k]

        return mat.to(device)

    def _get_cached_matrices(
        self, height: int, width: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(A_rows, A_cols, A_rows_pinv, A_cols_pinv)`` for the given HR size.

        Matrices are computed on first call for each unique ``(height, width, device)``
        combination and cached for subsequent calls.

        Args:
            height: HR spatial height.
            width:  HR spatial width.
            device: Target device.

        Returns:
            Tuple of four tensors:
            - ``A_rows``      — row downsample matrix, shape ``(H/s, H)``
            - ``A_cols``      — column downsample matrix, shape ``(W/s, W)``
            - ``A_rows_pinv`` — pseudoinverse of ``A_rows``, shape ``(H, H/s)``
            - ``A_cols_pinv`` — pseudoinverse of ``A_cols``, shape ``(W, W/s)``
        """
        cache_key = (height, width, str(device))

        if cache_key not in self._cache:
            A_rows = self._build_1d_downsample_matrix(height, device)
            A_cols = self._build_1d_downsample_matrix(width, device)

            A_rows_pinv = torch.linalg.pinv(A_rows)
            A_cols_pinv = torch.linalg.pinv(A_cols)

            self._cache[cache_key] = (A_rows, A_cols, A_rows_pinv, A_cols_pinv)
            logger.info(f"Cached BackProjection matrices for size {height}x{width}")

        return self._cache[cache_key]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x_lr: torch.Tensor, x_hr: torch.Tensor) -> torch.Tensor:
        """Apply back-projection correction.

        Adjusts ``x_hr`` so that ``downscale(output) ≈ x_lr``::

            output = x_hr + upscale(x_lr − downscale(x_hr))

        Args:
            x_lr: Low-resolution tensor, shape ``(B, C, h, w)``.
            x_hr: High-resolution estimate, shape ``(B, C, h*s, w*s)``.

        Returns:
            Corrected HR tensor, same shape as ``x_hr``.
        """
        lr_reconstructed = self.downscale(x_hr)
        residual_lr = x_lr - lr_reconstructed
        correction_hr = self.upscale(residual_lr)
        return x_hr + correction_hr

    # ------------------------------------------------------------------
    # Separable up/downscale via matrix multiplication
    # ------------------------------------------------------------------

    def upscale(self, lr: torch.Tensor) -> torch.Tensor:
        """Upsample using the pseudoinverse of the downsample matrices.

        Applies ``A_rows⁺`` along rows and ``A_cols⁺`` along columns (separable).

        Args:
            lr: Low-resolution tensor, shape ``(B, C, h, w)``.

        Returns:
            Upsampled tensor, shape ``(B, C, h*s, w*s)``.
        """
        B, C, h, w = lr.shape
        H, W = self.factor * h, self.factor * w

        _, _, A_rows_pinv, A_cols_pinv = self._get_cached_matrices(H, W, lr.device)

        # A_rows_pinv: (H, h),  A_cols_pinv: (W, w)
        # 1. Upsample rows: (H, h) @ (B, C, h, w) -> (B, C, H, w)
        temp = torch.einsum('Hh, bchw -> bcHw', A_rows_pinv, lr)
        # 2. Upsample columns: (B, C, H, w) x (W, w) -> (B, C, H, W)
        return torch.einsum('bcHw, Ww -> bcHW', temp, A_cols_pinv)

    def downscale(self, hr: torch.Tensor) -> torch.Tensor:
        """Downsample using the bicubic Toeplitz matrices.

        Applies ``A_rows`` along rows and ``A_cols`` along columns (separable).

        Args:
            hr: High-resolution tensor, shape ``(B, C, H, W)``.

        Returns:
            Downsampled tensor, shape ``(B, C, H/s, W/s)``.
        """
        B, C, H, W = hr.shape

        A_rows, A_cols, _, _ = self._get_cached_matrices(H, W, hr.device)

        # A_rows: (h, H),  A_cols: (w, W)
        # 1. Downsample rows: (h, H) @ (B, C, H, W) -> (B, C, h, W)
        temp = torch.einsum('hH, bcHW -> bchW', A_rows, hr)
        # 2. Downsample columns: (B, C, h, W) x (w, W) -> (B, C, h, w)
        return torch.einsum('bchW, wW -> bchw', temp, A_cols)