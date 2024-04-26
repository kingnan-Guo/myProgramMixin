# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         GradientTape
# Author:       mac
# Date:         2021/8/4
# Description:  
#
#
# -------------------------------------------------------------------------------

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = tf.constant(3.0)
with tf.GradientTape() as g:
    g.watch(x)
    y = x * x
    z = y * y
    # dy_dx = g.gradient(y, x)
    # print(dy_dx)
    dz_dx = g.gradient(z, x)
    print(dz_dx)






