import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_update_network(dest_net, src_net, tau):
    for o, n in zip(dest_net.parameters(), src_net.parameters()):
        o.data.copy_(tau * n.data + (1 - tau) * o.data)

def minibatch_inference(args, rollout_fn, batch_size=1000, cat_dim=0):
    data_size = len(args[0])
    num_batches = int(np.ceil(data_size / batch_size))
    inference_results = []
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min(data_size, (i + 1) * batch_size)
        input_batch = [ip[batch_start:batch_end] for ip in args]
        outputs = rollout_fn(*input_batch)
        if i == 0:
            if isinstance(outputs, tuple):
                multi_op = True
            else:
                multi_op = False
            inference_results = outputs
        else:
            if multi_op:
                inference_results = (torch.cat([prev_re, op], dim=cat_dim) for prev_re, op in
                                     zip(inference_results, outputs))
            else:
                inference_results = torch.cat([inference_results, outputs])
    return inference_results

def dict_batch_generator(data, batch_size, keys=None):
    if keys is None:
        keys = list(data.keys())
    num_data = len(data[keys[0]])
    num_batches = int(np.ceil(num_data / batch_size))
    indices = np.arange(num_data)
    np.random.shuffle(indices)
    for batch_id in range(num_batches):
        batch_start = batch_id * batch_size
        batch_end = min(num_data, (batch_id + 1) * batch_size)
        batch_data = {}
        for key in keys:
            batch_data[key] = data[key][indices[batch_start:batch_end]]
        yield batch_data
