"""
impr.py
Image Processing functions using scipy-style images (i.e., Numpy arrays of
width x height x channels floats on [0,1])
"""

import numpy as np

def paste(src, onto, loc=(0,0)):
  """
  Pastes one image onto another. The result will have the same size as the
  destination image, even if that means that part of the source image is
  cropped. The original destination image isn't modified; instead a copy is
  returned.

  https://stackoverflow.com/questions/28676187/numpy-blit-copy-part-of-an-array-to-another-one-with-a-different-size
  """
  result = np.copy(onto)
  i, j = loc
  nch = len(onto.shape - 2)
  ind = [
    slice(i, i + src.shape[0]),
    slice(j, j + src.shape[1]),
    slice(None)
  ]
  result[ind] = src[:src.shape[0] - i,:src.shape[1] - j]
  return result

def frame(img, size=2, color=(1., 1., 1.)):
  """
  Adds a frame to the given image, returning a larger image.
  """
  w, h, c = img.shape # TODO: w, h or h, w?
  result = np.pad(
    img,
    ((size, size), (size, size), (0, 0)),
    mode="constant",
    constant_values=0
  )
  result[:size,:,:] = color
  result[-size:,:,:] = color
  result[:,:size,:] = color
  result[:,-size:,:] = color

  return result

def concatenate(left, right, vert=False, pad_color=(1, 1, 1)):
  """
  Concatenates two images side-by-side. If "vert" is True they are arranged
  above (left) and below (right) each other instead. If the image sizes don't
  match, the second image is cropped or padded to match the size of the first.
  """
  if not vert and left.shape[0] != right.shape[0]:
    sd = left.shape[0] - right.shape[0]
    if sd > 0:
      right = np.pad(
        right,
        ((0, sd), (0, 0), (0, 0)),
        mode="constant",
        constant_values=0
      )
      right[-sd:,:,:] = pad_color
    else:
      right = right[:sd,:,:]
  elif vert and left.shape[1] != right.shape[1]:
    sd = left.shape[1] - right.shape[1]
    if sd > 0:
      right = np.pad(
        right,
        ((0, 0), (0, sd), (0, 0)),
        mode="constant",
        constant_values=0
      )
      right[:,-sd:,:] = pad_color
    else:
      right = right[:,:sd,:]

  return np.concatenate((left, right), axis=1 - int(vert))

def join(images, vert=False, pad_color=(1., 1., 1.)):
  """
  Works like concatenate, but accepts more than two images and builds either a
  horizontal or vertical line out of all of them.
  """
  if len(images) == 1:
    return images[0]

  stripe = images[0]
  stripe = concatenate(stripe, images[1], vert=vert, pad_color=pad_color)

  for i in range(2,len(images)):
    stripe = concatenate(stripe, images[i], vert=vert, pad_color=pad_color)

  return stripe

def montage(images, padding=2, pad_color=(1., 1., 1.)):
  """
  Generates a montage of the given images, adding padding to each beforehand if
  padding isn't zero (or None or False). Attempts to fit things into as square
  a shape as possible.
  """
  if padding:
    images = [frame(img, size=padding, color=pad_color) for img in miages]

  sqw = int(math.ceil(len(images)**0.5))
  sqh = sqw
  while len(images) <= sqw * sqh - sqw:
    sqh -= 1

  rows = []
  idx = 0
  for i in range(sqh):
    rows.append([])
    for j in range(sqw):
      rows[-1].append(images[idx])
      idx += 1
      if idx >= len(images):
        break

    if idx >= len(images):
      break

  rowimages = []
  for r in rows:
    rowimages.append(join(r, vert=False, pad_color=pad_color))

  return join(
    [ join(r, vert=False, pad_color=pad_color) for r in rows ],
    vert=True,
    pad_color=pad_color
  )
