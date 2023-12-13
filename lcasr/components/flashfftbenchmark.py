import torch
from flashfftconv.depthwise_1d import conv1dFunc
from torch.profiler import profile, record_function, ProfilerActivity   


def conv1d_test():
    B, N, D = 16, 1024, 768
    x = torch.randn(B, D, N, dtype=torch.float32, device='cuda')

    conv1d_torch = torch.nn.Conv1d(
        in_channels = D,
        out_channels = D,
        kernel_size = 9,
        groups = D,
        padding = 4,
        dtype = torch.bfloat16,
        device = 'cuda'
    )

    print(x.device.type)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                y_fftconv = conv1dFunc.apply(
                    x.contiguous(),
                    conv1d_torch.weight.squeeze(),
                    conv1d_torch.bias,
                    4,
                    True
                )

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                y_torch = conv1d_torch(x)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print(y_fftconv.shape, y_torch.shape)
    print(y_fftconv.dtype, y_torch.dtype)


if __name__ == "__main__":
    conv1d_test()
