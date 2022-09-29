
import torch
from torch.autograd import Variable

dtype = torch.FloatTensor

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.



def normalize(A,M):
    for i in range(M):
        A[:,i] = A[:,i] / (torch.sqrt(A[:,i].pow(2).sum()))
    return A
        

def gen_low_coherence(M,N,seed=None,silent=True):
    
    if seed is not None:
        torch.manual_seed(seed)
    # Create random Tensors for weights, and wrap them in Variables.
    # Setting requires_grad=True indicates that we want to compute gradients with
    # respect to these Variables during the backward pass.
    A = Variable(normalize(torch.randn(M, N).type(dtype),N), requires_grad=True)
    
    # Create identity matrix, target for A'A
    I = Variable(torch.eye(N).type(dtype), requires_grad=False)
    
    learning_rate = 1
    tot_iters = 1000
    for t in range(tot_iters):
        AtA = torch.mm(A.t(),A)
        loss = (AtA - I).pow(2).mean()
        
        if t==0:
            init_loss = loss
            
        if silent==False and t%100==0:
            print(t, loss.item())
    
        # Use autograd to compute the backward pass. This call will compute the
        # gradient of loss with respect to all Variables with requires_grad=True.
        loss.backward()
    
        # Update weights using gradient descent
        A.data -= learning_rate * A.grad.data
        A.data = normalize(A.data,N)
    
        # Manually zero the gradients after updating weights
        A.grad.data.zero_()
    
    print(f'Init loss = {init_loss} , final loss = {loss.item()}')
    
    return A.cpu().detach().numpy()