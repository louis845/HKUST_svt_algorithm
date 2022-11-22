import torch
import torch.nn
import utils
import scipy
import scipy.sparse
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GradientDescentMethod(torch.nn.Module):
    # initialize the network
    # we store row*rank (A) and rank*col (B) matrix.
    # A*B + row_bias + col_bias + global_bias = result
    # for a single result (r,c), 
    # result[r,c] = A[r,:]*B[:,c] + row_bias[r] + col_bias[c] + global_bias
    def __init__(self, row, col, rank):
        super().__init__()
        self.A = torch.nn.Embedding(row, rank, device = device)
        self.B = torch.nn.Embedding(col, rank, device = device)
        
        self.rows_total=row
        self.cols_total=col
    
    def calc_full(self, locs_row, locs_col):
        Asel = self.A(locs_row)
        Bsel = self.B(locs_col)
        
        return torch.einsum("ik,ik->i", Asel, Bsel)

    def forward(self, x):
        size = x.shape[1]
        Asel = self.A(x.select(0, 0))
        Bsel = self.B(x.select(0, 1))
        
        result = torch.einsum('ik,ik->i', Asel, Bsel)
        
        return result
    
    def construct_predicted_matrix(self):
        with torch.no_grad():
            print(self.A.weight.shape)
            print(self.B.weight.shape)
            
            res = torch.matmul(self.A.weight, self.B.weight.transpose(0,1))
            return res.cpu().detach().numpy()


# input: sparse matrix with given locations Omega, and the target rank.
def gradient_descent_completion(M, locations, rank, log = False, tolerance = 0.001):
    norm_of_m = scipy.sparse.linalg.norm(M, ord='fro')
    
    num_epochs = 10000
    batch_size = 24000
    
    num_samples = len(locations[0])
    # convert sparse matrix to a bunch of values
    values = []
    for k in range(num_samples):
        values.append(M[locations[0][k], locations[1][k]])
    
    network = GradientDescentMethod(M.shape[0],M.shape[1], rank)
    
    criterion = torch.nn.MSELoss(reduction='sum')
    optim = torch.optim.SGD(network.parameters(), lr=1e-5)
    t = 0
    
    tgt_vals = torch.from_numpy(np.array(values)).float().to(device)
    locs_row_tensor = torch.from_numpy(np.array(locations[0])).long().to(device)
    locs_col_tensor = torch.from_numpy(np.array(locations[1])).long().to(device)
    
    running = True
    while running:
        for batch in range( int(math.ceil((num_samples+0.0) / batch_size)) ):
            # obtain the cols, rows, and vals to be used.
            bstart = batch * batch_size
            bend = min((batch+1) * batch_size, num_samples)
            cols = locations[0][bstart:bend]
            rows = locations[1][bstart:bend]
            vals = values[bstart:bend]
            
            # convert to pytorch data
            locs = torch.from_numpy(np.array(
                [cols, rows]
            )).long().to(device)
            actual_vals = torch.from_numpy(np.array(vals)).float().to(device)
            
            # actual_vals = torch.tensor(vals, device = device)
            
            # eval and autograd
            pred_vals = network(locs)
            loss = criterion(pred_vals, actual_vals)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
        t = t + 1
        with torch.no_grad():
            ncten = network.calc_full(locs_row_tensor, locs_col_tensor)
            loss = criterion(ncten, tgt_vals)
            loss = math.sqrt(loss.item()) # sqrt of the MSE loss is the Frobenius norm
            relative_loss = loss / norm_of_m
            
            if log:
                if t % 40 == 0:
                    print("epoch: ",t,"  loss: ", loss, "  quotient loss: ", relative_loss)
            
            if relative_loss <= tolerance:
                running = False
        
    return network.construct_predicted_matrix()