import torch

class E(torch.autograd.Function):
    # y: The input tensor. In super-resolution, this is the model's output after processing the LR image

    @staticmethod
    def loss_fn(y, net):
        # net: The model to be used as the second application (frozen copy or original).
        loss = torch.mean((net(y) - y)**2)
        return loss

    @staticmethod
    def forward(ctx, y, net):
        # Save the input tensor 'y' for use in the backward pass
        ctx.save_for_backward(y)
        # Store the network reference in the context
        ctx.net = net
        return E.loss_fn(y, net)

    @staticmethod
    def backward(ctx, grad_output):
        """
        The backward pass utilizing the custom perturbation formula instead of 
        standard analytical gradients.
        
        Args:
            ctx: Context object containing saved tensors and network.
            grad_output (Tensor): The gradient flowing back from the previous operation.
            
        Returns:
            Tuple[Tensor, None]: The custom gradient with respect to 'y', and None 
                                 for 'net' (since we do not update this network's weights here).
        """
        # Retrieve the saved tensor and network from the forward pass
        y, = ctx.saved_tensors
        net = ctx.net
        
        # Perform the internal forward passes required for the perturbation formula
        y2 = net(y)
        y3 = net(y2)
        
        # Calculate the Modified Backpropagation gradient vector
        e = 3 * y2 - 2 * y3 - y
        
        # Scale the custom gradient by the batch size.
        # This is necessary because torch.mean() was used in the loss function, 
        # which averages over the batch dimension.
        grads = -e / e.shape[0]
        
        # Multiply by the incoming gradient (chain rule).
        # We return None for the 'net' argument because we do not want to compute 
        # gradients for the frozen network's parameters in this step.
        return grads * grad_output, None


class ELoss(torch.nn.Module):
    """
    A PyTorch Module wrapper for the custom autograd function E.
    This allows the custom loss to be easily integrated into standard PyTorch training loops.
    """
    def __init__(self, net, mode=None):
        """
        Args:
            net (nn.Module): The network to be used in the loss calculation 
                             (active or frozen, depending on the loss term).
            mode (str, optional): An optional string to define specific modes if needed.
        """
        super(ELoss, self).__init__()
        self.net = net
        self.mode = mode

    def forward(self, y):
        """
        Applies the custom autograd function to the input tensor.
        
        Args:
            y (Tensor): The tensor to calculate the loss for.
            
        Returns:
            Tensor: The calculated loss with the custom backward hook attached.
        """
        # Call the custom autograd function's apply method
        return E.apply(y, self.net)