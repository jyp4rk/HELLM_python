# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import transformers

from train_utils import apply_r3_r4, rtn_utils
from src.utils import fuse_norm_utils, hadamard_utils, quant_utils, utils


def prepare_model(args, model):
    transformers.set_seed(args.seed)
    model.eval()

    # Rotate the weights
    fuse_norm_utils.fuse_layer_norms(model)
    utils.cleanup_memory(verbos=True)

    quant_utils.add_actquant(model)  # Add Activation Wrapper to the model
    qlayers = quant_utils.find_qlayers(model)
    for name in qlayers:
        if "down_proj" in name:
            had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
            qlayers[name].online_full_had = True
            qlayers[name].had_K = had_K
            qlayers[name].K = K
            qlayers[name].fp32_had = args.fp32_had
    return model
