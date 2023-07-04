# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
This contains implementations of nwot based on the updated version of 
https://github.com/BayesWatch/nas-without-training
to reflect the second version of the paper https://arxiv.org/abs/2006.04647
"""

import torch
import numpy as np

from . import measure


@measure("nwot", bn=True)
def compute_nwot(net, inputs, targets, split_data=1, loss_fn=None, normalize=False, div_by_relu=False):

    batch_size = len(targets)
   
    def counting_forward_hook(module, inp, out):
        if hasattr(module, 'visited_backwards') and not module.visited_backwards:
            return

        inp = inp[0].view(inp[0].size(0), -1)
        x = (inp > 0).float() # binary indicator 
        K = x @ x.t() 
        K2 = (1.-x) @ (1.-x.t())

        if normalize:
            K /= x.shape[1]
            K2 /= x.shape[1]

        net.K = net.K + K.cpu().numpy() + K2.cpu().numpy() # hamming distance
        net.nrelu += 1

                
    def counting_backward_hook(module, inp, out):
        if out[0] is None or inp[0] is None:
            return

        all_zero = torch.eq(out[0], 0.0).all() or torch.eq(inp[0], 0.0).all()
        #print(all_zero)
        if not all_zero:
            module.visited_backwards = True

    net.nrelu = 0
    net.K = np.zeros((batch_size, batch_size))
    for name, module in net.named_modules():
        module_type = str(type(module))
        if ('ReLU' in module_type)  and ('naslib' not in module_type):
            module.naame = name
            module.register_full_backward_hook(counting_backward_hook)
            module.register_forward_hook(counting_forward_hook)

    x = torch.clone(inputs)
    y = net(x)

    s, jc = np.linalg.slogdet(net.K)
    first_val = jc

    for name, module in net.named_modules():
        module_type = str(type(module))
        if ('ReLU' in module_type)  and ('naslib' not in module_type):
            module.visited_backwards = False

    y.backward(torch.ones_like(y))
    net.nrelu = 0
    net.K = np.zeros((batch_size, batch_size))
    net(x)

    if div_by_relu:
        net.K /= net.nrelu

    s, second_val = np.linalg.slogdet(net.K)

    print(first_val, second_val)

    return second_val