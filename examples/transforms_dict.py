# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
import random

import logging
import numpy as np
import scipy
import scipy.ndimage
import scipy.interpolate
import torch
import collections
from scipy.linalg import expm, norm
# import MinkowskiEngine as ME


class RandomDropout(object):
  def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.2):
    """
    upright_axis: axis index among x,y,z, i.e. 2 for z
    """
    self.dropout_ratio = dropout_ratio
    self.dropout_application_ratio = dropout_application_ratio

  def __call__(self, in_dict):
    if random.random() < self.dropout_application_ratio:
      N = len(in_dict['pts'])
      inds = torch.randperm(N)[:int(N * (1 - self.dropout_ratio))]
      for k, v in in_dict.items():
        if isinstance(v, torch.Tensor):
          in_dict[k] = v[inds]
    return in_dict

class RandomHorizontalFlip(object):
  def __init__(self, upright_axis, aug_prob=0.95):
    """
    upright_axis: axis index among x,y,z, i.e. 2 for z
    """
    self.D = 3
    self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
    # Use the rest of axes for flipping.
    self.horz_axes = set(range(self.D)) - set([self.upright_axis])
    self.aug_prob = aug_prob

  def __call__(self, in_dict):
    if random.random() < self.aug_prob:
      for curr_ax in self.horz_axes:
        if random.random() < 0.5:
          coord_max = torch.max(in_dict['pts'])
          in_dict['pts'][:, curr_ax] = coord_max - in_dict['pts'][:, curr_ax]
          if 'normals' in in_dict:
            in_dict['normals'][:, curr_ax] = -in_dict['normals'][:, curr_ax]
          
    return in_dict

class RandomScaling(object):
  def __init__(self, scale_augmentation_min, scale_augmentation_max):
    self.scale_augmentation_min = scale_augmentation_min
    self.scale_augmentation_max = scale_augmentation_max

  def __call__(self, in_dict):
    scale = torch.rand(1, dtype=torch.float32) * (self.scale_augmentation_max - self.scale_augmentation_min) + self.scale_augmentation_min
    in_dict['pts'] *= scale
    return in_dict

def M(axis, theta):
  return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

class RandomRotation(object):
  def __init__(self, rotation_augmentation_bound):
    self.rotation_augmentation_bound = rotation_augmentation_bound

  def __call__(self, in_dict):
    if isinstance(self.rotation_augmentation_bound, collections.Iterable):
      rot_mats = []
      for axis_ind, rot_bound in enumerate(self.rotation_augmentation_bound):
        theta = 0
        axis = np.zeros(3)
        axis[axis_ind] = 1
        if rot_bound is not None:
          theta = np.random.uniform(*rot_bound)
        rot_mats.append(M(axis, theta))
      # Use random order
      np.random.shuffle(rot_mats)
      rot_mat = torch.tensor(rot_mats[0] @ rot_mats[1] @ rot_mats[2], dtype=torch.float32)
    else:
      raise ValueError()

    in_dict['pts'] = in_dict['pts'] @ rot_mat.T
    if 'normals' in in_dict:
      in_dict['normals'] = in_dict['normals']  @ rot_mat.T

    return in_dict

class PositionJitter(object):

  def __init__(self, std=0.01, aug_prob=0.95):
    self.std = std
    self.aug_prob=aug_prob

  def __call__(self, in_dict):
    # print("pos jitter")
    if random.random() < self.aug_prob:
      noise = torch.randn_like(in_dict['pts'])
      in_dict['pts'] += noise * self.std
      # print(noise)
    return in_dict


class ChromaticTranslation(object):
  """Add random color to the image, input must be an array in [0,255] or a PIL image"""

  def __init__(self, trans_range_ratio=1e-1, aug_prob=0.95):
    """
    trans_range_ratio: ratio of translation i.e. 255 * 2 * ratio * rand(-0.5, 0.5)
    """
    self.trans_range_ratio = trans_range_ratio
    self.aug_prob = aug_prob

  def __call__(self, in_dict):
    if 'colors' in in_dict:
      if random.random() < self.aug_prob:
        tr = (torch.rand(1, 3) - 0.5) * 255 * 2 * self.trans_range_ratio
        in_dict['colors'] = torch.clip(tr + in_dict['colors'], 0, 255)
    return in_dict


class ChromaticAutoContrast(object):
  def __init__(self, randomize_blend_factor=True, blend_factor=0.5):
    self.randomize_blend_factor = randomize_blend_factor
    self.blend_factor = blend_factor

  def __call__(self, in_dict):
    if 'colors' in in_dict:
      if random.random() < 0.2:
        # mean = np.mean(feats, 0, keepdims=True)
        # std = np.std(feats, 0, keepdims=True)
        # lo = mean - std
        # hi = mean + std
        lo = in_dict['colors'].min(0, keepdims=True)[0]
        hi = in_dict['colors'].max(0, keepdims=True)[0]

        scale = 255 / (hi - lo)

        contrast_feats = (in_dict['colors'] - lo) * scale

        blend_factor = random.random() if self.randomize_blend_factor else self.blend_factor
        in_dict['colors'] = (1 - blend_factor) * in_dict['colors'] + blend_factor * contrast_feats
    return in_dict


class ChromaticJitter(object):

  def __init__(self, std=0.01):
    self.std = std

  def __call__(self, in_dict):
    if 'colors' in in_dict:
      if random.random() < 0.95:
        noise = torch.randn_like(in_dict['colors'])
        noise *= self.std * 255
        in_dict['colors'] = torch.clip(noise + in_dict['colors'], 0, 255)
    return in_dict


class RGBtoHSV(object):

  def __init__(self):
    pass

  def __call__(self, in_dict):
    if 'colors' in in_dict:
      # print("hsv")
      # Translated from source of colorsys.rgb_to_hsv
      # r,g,b should be a numpy arrays with values between 0 and 255
      # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
      rgb = in_dict['colors'].float().numpy()
      hsv = np.zeros_like(rgb)

      r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
      maxc = np.max(rgb[..., :3], axis=-1)
      minc = np.min(rgb[..., :3], axis=-1)
      hsv[..., 2] = maxc
      mask = maxc != minc
      hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
      rc = np.zeros_like(r)
      gc = np.zeros_like(g)
      bc = np.zeros_like(b)
      rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
      gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
      bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
      hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
      hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
      in_dict['colors'] = torch.from_numpy(hsv)
    return in_dict


class Compose(object):
  """Composes several transforms together."""

  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, args):
    for t in self.transforms:
      args = t(args)
    return args


class ElasticDistortion(object):
  def __init__(self, granularity=0.2, magnitude=0.4):
    self.granularity = granularity
    self.magnitude = magnitude

  def __call__(self, in_dict):
    """Apply elastic distortion on sparse coordinate space.

      pointcloud: numpy array of (number of points, at least 3 spatial dims)
      granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
      magnitude: noise multiplier
    """
    if random.random() < 0.95:
      blurx = torch.ones((3, 1, 1, 1), dtype=torch.float32) / 3
      blury = torch.ones((1, 3, 1, 1), dtype=torch.float32) / 3
      blurz = torch.ones((1, 1, 3, 1), dtype=torch.float32) / 3
      coords = in_dict['pts']
      coords_min = coords.min(0)[0]

      # Create Gaussian noise tensor of the size given by granularity.
      noise_dim = ((coords - coords_min).max(0)[0] // self.granularity).int() + 3
      noise = torch.randn(*noise_dim, 3, dtype=torch.float32)

      # Smoothing.
      for _ in range(2):
        noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

      # Trilinear interpolate noise filters for each spatial dimensions.
      ax = [
          torch.linspace(d_min, d_max, d)
          for d_min, d_max, d in zip(coords_min - self.granularity, coords_min + self.granularity *
                                     (noise_dim - 2), noise_dim)
      ]
      interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
      in_dict['pts'] = coords + interp(coords) * self.magnitude
    return in_dict
