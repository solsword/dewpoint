"""
impr.py
Image Processing functions using scipy-style images (i.e., Numpy arrays of
width x height x channels floats on [0,1])
"""

import math

import numpy as np

from PIL import Image, ImageDraw, ImageFont

from skimage import img_as_float

def blank(wh, color=(0., 0., 0.)):
  """
  Returns a new blank image of the given width and height.
  """
  w, h = wh
  result = np.zeros((h, w, 3))
  for i in range(result.shape[0]):
    for j in range(result.shape[1]):
      result[i,j,:] = color

  return result

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

def frame(img, size=2, color=(0., 0., 0.)):
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

def pad_crop(image, w=None, h=None, center=True, pad_color=(0., 0., 0.)):
  """
  Pads and/or crops the image to the given width and height. If either is not
  given, the image's original dimension is used. If center is given, the image
  center will gravitate to the center of the new size, otherwise the image
  origin will be aligned with the new origin.
  """
  if w is None:
    w = image.shape[1]
  if h is None:
    h = image.shape[0]

  result = np.copy(image)

  vd = image.shape[0] - h
  if vd > 0: # cropping height
    if center:
      half = vd//2
      result = result[half:-(vd - half),:,:]
    else:
      result = result[:-vd,:,:]
  elif vd < 0: # padding height
    if center:
      half = vd//2
      result = np.pad(
        result,
        ((-half, -(vd - half)), (0, 0), (0, 0)),
        mode="constant",
        constant_values=0
      )
      result[:-half,:,:] = pad_color
      result[(vd - half):,:,:] = pad_color
    else:
      result = np.pad(
        result,
        ((0, -vd), (0, 0), (0, 0)),
        mode="constant",
        constant_values=0
      )
      result[vd:,:,:] = pad_color

  hd = image.shape[1] - w
  if hd > 0: # cropping width
    if center:
      half = hd//2
      result = result[:,half:-(hd - half),:]
    else:
      result = result[:,:-hd,:]
  elif hd < 0: # padding width
    if center:
      half = hd//2
      result = np.pad(
        result,
        ((0, 0), (-half, -(hd - half)), (0, 0)),
        mode="constant",
        constant_values=0
      )
      result[:,hd - half:,:] = pad_color
      result[:,:-half,:] = pad_color
    else:
      result = np.pad(
        result,
        ((0, 0), (0, -hd), (0, 0)),
        mode="constant",
        constant_values=0
      )
      result[:,hd:,:] = pad_color

  return result

def concatenate(left, right, vertical=False, center=True, pad_color=(0., 0., 0.)):
  """
  Concatenates two images side-by-side. If "vertical" is True they are arranged
  above (left) and below (right) each other instead. If the image sizes don't
  match, the smaller image is padded to match the size of the larger, with
  padding filled using the given "pad_color". If "center" is given, smaller
  images are centered relative to larger ones.
  """
  if vertical:
    if left.shape[1] > right.shape[1]:
      right = pad_crop(right, left.shape[1], None)
    elif left.shape[1] < right.shape[1]:
      left = pad_crop(left, right.shape[1], None)
  else:
    if left.shape[0] > right.shape[0]:
      right = pad_crop(right, None, left.shape[0])
    elif left.shape[0] < right.shape[0]:
      left = pad_crop(left, None, right.shape[0])

  return np.concatenate((left, right), axis=1 - int(vertical))

def join(
  images,
  vertical=False,
  center=True,
  padding=0,
  pad_color=(0., 0., 0.)
):
  """
  Works like concatenate, but accepts more than two images and builds either a
  horizontal or vertical line out of all of them. Images that are too small in
  the non-joined dimension are either aligned at one edge, or center-aligned if
  "center" is given, and any blank space left over is filled with the given pad
  color. If "padding" is given each image will be framed first.
  """
  if len(images) == 1:
    return images[0]

  if padding:
    images = [frame(img, size=padding, color=pad_color) for img in images]

  stripe = images[0]
  stripe = concatenate(
    stripe,
    images[1],
    vertical=vertical,
    center=center,
    pad_color=pad_color
  )

  for i in range(2,len(images)):
    stripe = concatenate(
      stripe,
      images[i],
      vertical=vertical,
      center=center,
      pad_color=pad_color
    )

  return stripe

def montage(
  images,
  padding=2,
  pad_color=(0., 0., 0.),
  labels=None,
  label_color=(1, 1, 1)
):
  """
  Generates a montage of the given images, adding padding to each beforehand if
  padding isn't zero (or None or False) and labelling each if labels are given,
  using text in the given label_color. Attempts to fit things into as square a
  shape as possible.
  """
  if labels:
    images = [
      labeled(img, lbl, text=label_color, background=pad_color)
        for img, lbl in zip(images, labels)
    ]

  if padding:
    images = [ frame(img, size=padding, color=pad_color) for img in images ]

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
    rowimages.append(join(r, vertical=False, pad_color=pad_color))

  return join(
    [ join(r, vertical=False, pad_color=pad_color) for r in rows ],
    vertical=True,
    pad_color=pad_color
  )

def text_image(
  txt,
  size=12,
  foreground=(1., 1., 1.),
  background=(0., 0., 0.)
):
  """
  Creates and returns a new image array containing pixel values corresponding
  to the given text rendered in the foreground color on the background color,
  in the given font size.
  """
  fg = tuple(int(255 * f) for f in foreground)
  bg = tuple(int(255 * b) for b in background)

  blank = Image.new("RGB", (1, 1))
  try:
    fnt = ImageFont.truetype("DejaVuSansMono.ttf", size)
  except:
    fnt = ImageFont.load_default()
  tdr = ImageDraw.Draw(blank)

  # Compute text size using blank canvas:
  cw, _ = tdr.textsize("_", font=fnt)
  tw, th = tdr.textsize(txt, font=fnt)

  hpad = min(tw*0.2, 1.8*cw)
  vpad = 0.5 * th

  baseline = int(vpad/2)

  cw = int(tw + hpad)
  ch = int(th + vpad)

  canvas = Image.new("RGB", (cw, ch), bg)
  dr = ImageDraw.Draw(canvas)
  dr.text((int(hpad/2), baseline), txt, font=fnt, fill=fg)

  result = np.array(canvas, np.uint8)

  result = img_as_float(result)

  return result

def labeled(image, label, text=(1., 1., 1.), background=(0., 0., 0.)):
  """
  Returns an image which adds the given label text below the given image.
  """
  return concatenate(
    image,
    text_image(label, size=12, foreground=text, background=background),
    vertical=True,
    center=True,
    pad_color=background
  )
