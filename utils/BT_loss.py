import torch

# calls all the helper functions to calculate the loss
def BT_loss(A, B, loss_lambda):
    ''' A and B are the two represenations of augmented Data.
    loss_lambda is the regularization parameter.
    Returns the on_diag, off_diag and total loss'''
    A_norm, B_norm = normalize(A), normalize(B)
    #print(A_norm, B_norm)
    C = cross_corr_matrix(A_norm, B_norm, A.shape[0])
    on_diag, off_diag, loss = cross_corr_matrix_loss(C, loss_lambda)
    return on_diag, off_diag, loss


# Helper functions
def cross_corr_matrix_loss(C, lambd):
    '''C is the cross-correlation matrix. lambd is the regularization parameter.
    Compute the loss for the cross-correlation matrix by comapring it with the identity matrix'''
    on_diag = lambd[0] * torch.pow(get_on_diag(C) - torch.eye(C.shape[0], device=C.device), 2)
    off_diag = lambd[1] * torch.pow(get_off_diag(C), 2)
    #print(on_diag, off_diag)
    loss = torch.sum(on_diag) + torch.sum(off_diag)
    return torch.sum(on_diag), torch.sum(off_diag), loss

def get_off_diag(C):
    '''Returns the off-diagonal elements of the matrix'''
    # Makes the diagonal elements 0
    mask = torch.ones(C.shape, device=C.device)  # Use the same device as the tensor
    mask -= torch.eye(C.shape[0], device=C.device)  # Creates a mask to keep non-diagonal elements
    return C * mask  # Applies the mask to get only off-diagonal elements

def get_on_diag(C):
    '''Returns the diagonal elements of the matrix'''
    mask = torch.eye(C.shape[0], device=C.device)  # Use the same device as the tensor
    return C * mask  # Applies the mask to get only off-diagonal elements

# normalize across the columns
def normalize(C):
    '''Normalize the columns of the matrix'''
    C = C - torch.mean(C, dim=0, keepdim=True)
    C = C / torch.std(C, dim=0, keepdim=True)
    return C

def cross_corr_matrix(A, B, batch_size):
    '''A and B are the two represenations of augmented Data.
    batch_size is the number of samples in the batch.
    Returns the cross-correlation matrix of A and B'''
    C = torch.matmul(torch.t(A), B) / batch_size
    #print(C)
    return C


