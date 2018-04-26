# Exercise resolution

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
