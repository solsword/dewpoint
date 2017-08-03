#!/usr/bin/env python3

import impr

from matplotlib import pyplot as plt

def test():
  t = impr.text_image("hello, world")
  plt.imshow(t)
  plt.show()

  b1 = impr.blank((100, 20), color=(1., 1., 1.))
  b2 = impr.blank((30, 40), color=(0.5, 1., 1.))

  pc = impr.pad_crop(b1, w=50, h=50, pad_color=(0.0, 0.5, 0.5))

  plt.imshow(pc)
  plt.show()

  pc = impr.pad_crop(b2, w=50, h=50, pad_color=(0.0, 0.5, 0.5))

  plt.imshow(pc)
  plt.show()

  comb = impr.concatenate(b1, b2)

  plt.imshow(comb)
  plt.show()

  together = impr.join([b1, b2, t])
  together = impr.concatenate(together, t, vert=True)

  plt.imshow(together)
  plt.show()

if __name__ == "__main__":
  test()
