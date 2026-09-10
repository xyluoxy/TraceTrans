"""TraceTrans: joint image translation and spatial tracing for surgical prediction.

Reference implementation of the model from the paper
"TraceTrans: Translation and Spatial Tracing for Surgical Prediction"
(https://arxiv.org/abs/2510.22379).

Submodules:
    data_loaders    -- image-to-image training data generators.
    layers          -- warping / integration primitives and the two-decoder U-Net.
    losses          -- similarity, smoothness and adversarial loss terms.
    networks        -- the DualStreamBackbone generator and CycleGAN discriminators.
    tracetrans_model -- TraceTransModel, the training / inference orchestrator.
"""

import os
import torch

import neurite


from . import data_loaders
from . import layers
from . import losses
from . import networks
from . import tracetrans_model