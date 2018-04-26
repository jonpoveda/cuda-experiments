# Exercise resolutions

## [Master18-M5-Ex1]

Wording:

> Find a configuration of parameters that produces a scenario in which
cuDNN would select Winograd as its best choice to run a convolution
in the resulting scenario.

As explained in [documentation][cuda-doc] and in `cudnn.h` used, we 
expect to execute `convolution.py` and get the algorithm number 6 or 
7, refering to `CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD` or 
`CUDNN_CONVOLUTION_FWD_ALGO_​WINOGRAD_NONFUSED`. Just running the script 
we get a 5 meaning `CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING` is selected 
as the best option.

Default configuration is the following:
```python
n_input = 64
filters_in = 128
filters_out = 128
height_in = 112
width_in = 112
height_filter = 7
width_filter = 7
pad_h = 3
pad_w = 3
vertical_stride = 1
horizontal_stride = 1
upscalex = 1
upscaley = 1
alpha = 1.0
beta = 1.0
```

Running it several times with the default parameters we get an average 
of 28.1 seconds.

Knowing Winograd outperforms FFT on small filters, we replace the 7x7 
filters by 3x3. In the configuration:

```python
height_filter = 3
width_filter = 3
```

And effectively, the script returns 
`CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD` as the algorithm to use with a 
average time of 18.2 s.


## Annex

Extract of the `cudnn.h` used:

```cpp
typedef enum
{
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM = 2,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = 3,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT = 4,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = 5,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = 6,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = 7,
    CUDNN_CONVOLUTION_FWD_ALGO_COUNT = 8,
} cudnnConvolutionFwdAlgo_t;
```

## [Master18-M5-Ex2]

Wording:

> Complete the code to produce successive convolutions using cuDNN 
based on an initial result which is already done. You will find further 
instructions in the C/C++ code. All you should need is to have a glance 
at the code and check cuDNN SDK documentation.

The solution is to replace the `checkCUDNN` call inside the specified 
section by this one:

```cpp
checkCUDNN(cudnnConvolutionForward(cudnn,
                                   &alpha,
                                   input_descriptor,
                                   d_input,
                                   kernel_descriptor,
                                   d_new_kernel,
                                   convolution_descriptor,
                                   convolution_algorithm,
                                   d_workspace,
                                   workspace_bytes,
                                   &beta,
                                   output_descriptor,
                                   d_output));
```

The full file can be seen in [conv2_ex2.cu](conv/conv_ex2.cu)


[cuda-doc]: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionFwdAlgo_t
