import torch

last_use_cuda = True

def cuda(tensor, use_cuda = None):
    """
    A cuda wrapper
    """
    global last_use_cuda
    if use_cuda == None:
        use_cuda = last_use_cuda
    last_use_cuda = use_cuda
    if not use_cuda:
        return tensor
    if tensor is None:
        return None
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor

def sequence_mask(sequence_length, max_length=None):
    """
    e.g., sequence_length = "5,7,8", max_length=None
    it will return
    tensor([[ 1,  1,  1,  1,  1,  0,  0,  0],
            [ 1,  1,  1,  1,  1,  1,  1,  0],
            [ 1,  1,  1,  1,  1,  1,  1,  1]], dtype=torch.float32)
    :param sequence_length: a torch tensor
    :param max_length: if not given, it will be set to the maximum of `sequence_length`
    :return: a tensor with dimension  [*sequence_length.size(), max_length]
    """
    if len(sequence_length.size()) > 1:
        ori_shape = list(sequence_length.size())
        sequence_length = sequence_length.view(-1) # [N, ?]
        reshape_back = True
    else:
        reshape_back = False

    if max_length is None:
        max_length = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_length).long() # [max_length]
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_length) # [batch, max_len], repeats on each column
    seq_range_expand = torch.autograd.Variable(seq_range_expand).to(sequence_length.device)
    #if sequence_length.is_cuda:
    #    seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand) # [batch, max_len], repeats on each row

    ret = (seq_range_expand < seq_length_expand).float() # [batch, max_len]

    if reshape_back:
        ret = ret.view(ori_shape + [max_length])

    return ret

def MSE(src, dest, dim = None):
    res = src - dest
    res = res * res
    if dim == None:
        return res.mean()
    return res.mean(dim = dim)
