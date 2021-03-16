import torch
import numpy as np


fp32_mask = [0,
    0x00400000, 0x00600000, 0x00700000, 0x00780000,
    0x007c0000, 0x007e0000, 0x007f0000, 0x007f8000,
    0x007fc000, 0x007fe000, 0x007ff000, 0x007ff800,
    0x007ffc00, 0x007ffe00, 0x007fff00, 0x007fff80,
    0x007fffc0, 0x007fffe0, 0x007ffff0, 0x007ffff8, 0x007fffff]

fp64_mask = [0,
    0x0040000000000000, 0x0060000000000000, 0x0070000000000000, 0x0078000000000000,
    0x007c000000000000, 0x007e000000000000, 0x007f000000000000, 0x007f800000000000,
    0x007fc00000000000, 0x007fe00000000000, 0x007ff00000000000, 0x007ff80000000000,
    0x007ffc0000000000, 0x007ffe0000000000, 0x007fff0000000000, 0x007fff8000000000]


# set_mantissa_tensor : set to tensor or numpy array to speicific mantissa bits 
# TODO : Set direction of grouping
def set_mantissa_tensor(inp, group_mantissa):
    inp_n = inp.numpy() # inp_n = inp # For debug,
    # Convert to byte stream
    st = inp_n.tobytes() 
    # Set to uint32 array to easy computing
    v = np.frombuffer(st, dtype=np.uint32) 
    # Generate mask
    r_mask = np.asarray(np.full(v.shape, 0x007fffff, dtype=np.uint32))
    # Shift to make reversed mask
    r_mask = np.right_shift(r_mask, group_mantissa)
    # Get the reversed mask
    r_mask = np.invert(r_mask)
    # And operation to remove mantissa
    r_ = np.bitwise_and(v, r_mask)
    # revert to original np.float32 
    r = np.frombuffer(r_, dtype=np.float32)
    return torch.from_numpy(r.reshape(inp_n.shape))

# _make_group_tensor : Group values as same exponent bits, which shifts mantissa
# TODO : Set direction of grouping
def make_groups_tensor(inp, group_mantissa, group_size, group_direction):
    inp_n = inp.numpy() # inp_n = inp # For debug,
    # Transpose to replace direction
    inp_n = np.transpose(inp_n, group_direction) # (2,3,0,1)=kernel_input_output
    # After transposing, change array if first two dims are not 3x3
    # Save original dimention for later use
    orig_shape = inp_n.shape
    # Code modified from https://stackoverflow.com/questions/42297115/numpy-split-cube-into-cubes/42298440#42298440
    if inp_n.shape[0] % 3 != 0:
        # Pad Array to have more values
        inp_n = np.pad(inp_n, ((0,3-inp_n.shape[0]%3),(0,3-inp_n.shape[1]%3),(0,0),(0,0)))
        # Save padded size, which is used later
        padded_shape = inp_n.shape
        # Reshape to corresponding size to manipulate easier
        inp_n = inp_n.reshape([inp_n.shape[0]//3, 3, inp_n.shape[1]//3, 3, inp_n.shape[2], 1, inp_n.shape[3], 1])
        # Transpose to have 3x3 structure is preserved
        inp_n = inp_n.transpose([4,6,0,2,1,3,5,7])
        # Reshape array to (nx3x3x1x1)
        inp_n = inp_n.reshape(-1, 3, 3, 1, 1)
    # Convert to byte stream
    st = inp_n.tobytes() 
    # Set to uint32 array to easy computing
    v = np.frombuffer(st, dtype=np.uint32) 
    # Extract exponent
    e_mask = np.full(v.shape, 0x7f800000, dtype=np.uint32)
    e_ = np.bitwise_and(v, e_mask)
    # Get the max value
    # IDEA : send shift code to back, maybe that's faster
    np.right_shift(e_, 23, out=e_)
    # Match shape to divisible to group size
    m_ = np.append(e_, np.zeros(group_size - e_.shape[0] % group_size, dtype=np.uint32))
    m_ = np.reshape(m_, (group_size, -1))
    # get the max value of each blocks
    m_ = np.amax(m_, axis=0)
    # Revert back to original size
    m_ = np.repeat(m_, group_size)
    # Match shape back to input
    m_ = m_[:e_.shape[0]]
    # Difference of the exponent, -1 is applied because for more accurate hardware-wise simulation
    # On hardware, mantissa bits have to store the 1 from the IEEE Standard
    e_ = group_mantissa - (m_ - e_) - 1
    r_mask = np.full(v.shape, 0x007fffff, dtype=np.uint32)
    # When mantissa have to shift more than precision bits, total value have to be zero.
    r_mask[e_ > 0xff] = 0xffffffff
    # Clip the negative value
    # Maybe more smarter way...? -> np.clip(e_, 0, 0xff, out=e_)
    e_[e_ > 0xff] = 0
    # Shift to make reversed mask
    np.right_shift(r_mask, e_, out=r_mask)
    # Get the reversed mask
    np.invert(r_mask, out=r_mask)
    r_ = np.bitwise_and(v, r_mask)
    # revert to original np.float32 
    r = np.frombuffer(r_, dtype=np.float32)
    # revert back to original shape
    r = r.reshape(inp_n.shape)
    # Revert back change array if first two dims are not 3x3
    if orig_shape[0] %3 != 0:
        # Reshape into 3x3 size
        r = r.reshape([orig_shape[2], orig_shape[3], orig_shape[0]//3+1, orig_shape[1]//3+1, 3, 3, 1, 1])
        # Transpose to right order
        r = r.transpose([2,4,3,5,0,6,1,7])
        # Revert back to transposed shape
        r = r.reshape(padded_shape)
        # Cut the array
        r = r[:(orig_shape[0]%3)-3,:(orig_shape[1]%3)-3,:,:]
    return torch.from_numpy(np.transpose(r,group_direction))
