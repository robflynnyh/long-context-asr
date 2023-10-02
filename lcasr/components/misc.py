
def scale_grad(module, grad_input, grad_output, scale=1.0):
    for param in module.parameters(): param.grad *= scale