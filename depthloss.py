
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

class DepthSmoothLoss(nn.Module):
    """
    Compute the depth smoothness loss, defined as the weighted smoothness
    of the inverse depth.
    """

    def __init__(self, depth_max,eps,weight = 1.0):
        super().__init__()
        self.depth_max = depth_max
        self.eps = eps
        self.weight = weight

    def forward(self, images: Tensor, depths: Tensor) -> Tensor:
        if len(images) == 0:
            return depths.sum()

        mask = torch.ones_like(depths).bool()#doubts, mask must be c = 1, for images

        # compute inverse depths
        idepths = 1 / (depths / self.depth_max).clamp(self.eps)
        # idepths = depths.float() / self.depth_max
        # idepths = depths
        # idepths = 1 / self._nsb(depths, is_small=True)

        # compute the gradients
        idepth_dx: Tensor = self._gradient_x(idepths)
        idepth_dy: Tensor = self._gradient_y(idepths)
        image_dx: Tensor = self._gradient_x(images)
        image_dy: Tensor = self._gradient_y(images)

        # compute image weights
        weights_x: Tensor = torch.exp(-torch.mean(torch.abs(image_dx) + self.eps, dim=1, keepdim=True))
        weights_y: Tensor = torch.exp(-torch.mean(torch.abs(image_dy) + self.eps, dim=1, keepdim=True))

        # apply image weights to depth
        smoothness_x: Tensor = torch.abs(idepth_dx * weights_x)
        smoothness_y: Tensor = torch.abs(idepth_dy * weights_y)

        mask_x = mask[:, :, :, 1:]
        mask_y = mask[:, :, 1:, :]

        # loss for x and y
        loss_x = smoothness_x[mask_x].mean()
        loss_y = smoothness_y[mask_y].mean()

        assert loss_x.isfinite()
        assert loss_y.isfinite()

        return (loss_x + loss_y)*self.weight

    def _gradient_x(self, img: Tensor) -> Tensor:
        if len(img.shape) != 4:
            raise AssertionError(img.shape)
        return img[:, :, :, :-1] - img[:, :, :, 1:]

    def _gradient_y(self, img: Tensor) -> Tensor:
        if len(img.shape) != 4:
            raise AssertionError(img.shape)
        return img[:, :, :-1, :] - img[:, :, 1:, :]
    
class PanopticGuidedSmooth(nn.Module):
    """
    Compute the depth smoothness loss, defined as the weighted smoothness
    of the inverse depth.
    """

    def __init__(self, depth_max,eps,weight=1.0):
        super().__init__()
        self.padded = False
        self.depth_max = depth_max
        self.eps = eps
        self.weight = weight

    def forward(self, panoptic_truth: Tensor, depths: Tensor) -> Tensor:
        if len(panoptic_truth) == 0:
            return depths.sum()

        mask = torch.ones_like(panoptic_truth).bool()#doubts, mask must be c = 1, for images

        # compute inverse depths
        idepths = 1 / (depths / self.depth_max).clamp(self.eps)
        # idepths = depths.float() / self.depth_max
        # idepths = depths
        # idepths = 1 / self._nsb(depths, is_small=True)

        # compute the gradients
        idepth_dx: Tensor = self._gradient_x(idepths)
        idepth_dx = idepth_dx[:,:,:,:-1]
        idepth_dy: Tensor = self._gradient_y(idepths)
        idepth_dy = idepth_dy[:,:,:-1,:]
        panoptic_x: Tensor = self._panoptic_x(panoptic_truth).to(torch.float32)
        panoptic_y: Tensor = self._panoptic_y(panoptic_truth).to(torch.float32)
        panoptic_dx: Tensor = self._gradient_x(panoptic_x)
        panoptic_dy: Tensor = self._gradient_y(panoptic_y)

        # compute image weights
        weights_x: Tensor = 1 - panoptic_dx
        weights_y: Tensor = 1 - panoptic_dy

        # apply image weights to depth
        smoothness_x: Tensor = torch.abs(idepth_dx * weights_x)
        smoothness_y: Tensor = torch.abs(idepth_dy * weights_y)

        mask_x = mask[:, :, :, 2:]
        mask_y = mask[:, :, 2:, :]

        # loss for x and y
        loss_x = smoothness_x[mask_x].mean()
        loss_y = smoothness_y[mask_y].mean()

        assert loss_x.isfinite()
        assert loss_y.isfinite()

        return (loss_x + loss_y) * self.weight

    def _gradient_x(self, img: Tensor) -> Tensor:
        if len(img.shape) != 4:
            raise AssertionError(img.shape)
        return img[:, :, :, :-1] - img[:, :, :, 1:]

    def _gradient_y(self, img: Tensor) -> Tensor:
        if len(img.shape) != 4:
            raise AssertionError(img.shape)
        return img[:, :, :-1, :] - img[:, :, 1:, :]
    
    def _panoptic_x(self, img: Tensor) -> Tensor:
        if len(img.shape) != 4:
            raise AssertionError(img.shape)
        return img[:, :, :, :-1] != img[:, :, :, 1:]

    def _panoptic_y(self, img: Tensor) -> Tensor:
        if len(img.shape) != 4:
            raise AssertionError(img.shape)
        return img[:, :, :-1, :] != img[:, :, 1:, :]

class PanopticGuidedEdgeLoss(nn.Module):
    """
    Compute the depth smoothness loss, defined as the weighted smoothness
    of the inverse depth.
    """

    def __init__(self, depth_max,eps,weight = 1.0):
        super().__init__()
        self.padded = False
        self.depth_max = depth_max
        self.eps = eps
        self.weight = weight

    def forward(self, panoptic_truth: Tensor, depths: Tensor) -> Tensor:
        if len(panoptic_truth) == 0:
            return depths.sum()

        mask = torch.ones_like(panoptic_truth).bool()#doubts, mask must be c = 1, for images

        # compute inverse depths
        idepths = 1 / (depths / self.depth_max).clamp(self.eps)
        # idepths = depths.float() / self.depth_max
        # idepths = depths
        # idepths = 1 / self._nsb(depths, is_small=True)

        # compute the gradients
        idepth_dx: Tensor = self._gradient_x(idepths)
        idepth_dx = idepth_dx[:,:,:,:-1]
        idepth_dy: Tensor = self._gradient_y(idepths)
        idepth_dy = idepth_dy[:,:,:-1,:]
        panoptic_x: Tensor = self._panoptic_x(panoptic_truth).to(torch.int)
        panoptic_y: Tensor = self._panoptic_y(panoptic_truth).to(torch.int)
        panoptic_dx: Tensor = self._gradient_x(panoptic_x)
        panoptic_dy: Tensor = self._gradient_y(panoptic_y)

        # compute image weights
        weights_x: Tensor = torch.exp(-torch.mean(torch.abs(idepth_dx) + self.eps, dim=1, keepdim=True))
        weights_y: Tensor = torch.exp(-torch.mean(torch.abs(idepth_dy) + self.eps, dim=1, keepdim=True))

        # apply image weights to depth
        smoothness_x: Tensor = torch.abs(panoptic_dx * weights_x)
        smoothness_y: Tensor = torch.abs(panoptic_dy * weights_y)

        mask_x = mask[:, :, :, 2:]
        mask_y = mask[:, :, 2:, :]

        # loss for x and y
        loss_x = smoothness_x[mask_x].mean()
        loss_y = smoothness_y[mask_y].mean()

        assert loss_x.isfinite()
        assert loss_y.isfinite()

        return (loss_x + loss_y)*self.weight

    def _gradient_x(self, img: Tensor) -> Tensor:
        if len(img.shape) != 4:
            raise AssertionError(img.shape)
        return img[:, :, :, :-1] - img[:, :, :, 1:]

    def _gradient_y(self, img: Tensor) -> Tensor:
        if len(img.shape) != 4:
            raise AssertionError(img.shape)
        return img[:, :, :-1, :] - img[:, :, 1:, :]
    
    def _panoptic_x(self, img: Tensor) -> Tensor:
        if len(img.shape) != 4:
            raise AssertionError(img.shape)
        return img[:, :, :, :-1] != img[:, :, :, 1:]

    def _panoptic_y(self, img: Tensor) -> Tensor:
        if len(img.shape) != 4:
            raise AssertionError(img.shape)
        return img[:, :, :-1, :] != img[:, :, 1:, :]

if __name__ == "__main__":
    d = PanopticGuidedEdgeLoss(depth_max=89,eps=1e-6)
    img = torch.randn((14,1,400,698))
    de = torch.randn((14,1,400,698))
    # m = torch.randn((14,1,400,698))
    ft = d.forward(img,de,None)