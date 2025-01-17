import numpy as np
import torch
import torch.nn.functional as F#
from torch import nn
from torch.functional import F
from torch.optim import Adam
from torch_geometric.nn import MetaLayer, MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Softplus
from torch.autograd import Variable, grad
from torch_geometric.utils import add_self_loops, softmax

def make_packer(n, n_f):
    def pack(x):
        return x.reshape(-1, n_f*n)
    return pack

def make_unpacker(n, n_f):
    def unpack(x):
        return x.reshape(-1, n, n_f)
    return unpack

def get_edge_index(n, sim):
    if sim in ['string', 'string_ball']:
        #Should just be along it.
        top = torch.arange(0, n-1)
        bottom = torch.arange(1, n)
        edge_index = torch.cat(
            (torch.cat((top, bottom))[None],
             torch.cat((bottom, top))[None]), dim=0
        )
    else:
        adj = (np.ones((n, n)) - np.eye(n)).astype(int)
        edge_index = torch.from_numpy(np.array(np.where(adj)))

    return edge_index


class GN(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, hidden=300, aggr='add'):
        super(GN, self).__init__(aggr=aggr)  # "Add" aggregation.
        self.msg_fnc = Seq(
            Lin(2*n_f, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            ##(Can turn on or off this layer:)
#             Lin(hidden, hidden), 
#             ReLU(),
            Lin(hidden, msg_dim)
        )
        
        self.node_fnc = Seq(
            Lin(msg_dim+n_f, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
#             Lin(hidden, hidden),
#             ReLU(),
            Lin(hidden, ndim)
        )
    
    #[docs]
    def forward(self, x, edge_index):
        #x is [n, n_f]
        x = x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
      
    def message(self, x_i, x_j):
        # x_i has shape [n_e, n_f]; x_j has shape [n_e, n_f]
        tmp = torch.cat([x_i, x_j], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.msg_fnc(tmp)
    
    def update(self, aggr_out, x=None):
        # aggr_out has shape [n, msg_dim]

        tmp = torch.cat([x, aggr_out], dim=1)
        return self.node_fnc(tmp) #[n, nupdate]


class OGN(GN):
    def __init__(
		self, n_f, msg_dim, ndim, dt,
		edge_index, aggr='add', hidden=300, nt=1):

        super(OGN, self).__init__(n_f, msg_dim, ndim, hidden=hidden, aggr=aggr)
        self.dt = dt
        self.nt = nt
        self.edge_index = edge_index
        self.ndim = ndim
    
    def just_derivative(self, g, augment=False, augmentation=3):
        #x is [n, n_f]f
        x = g.x
        ndim = self.ndim
        if augment:
            augmentation = torch.randn(1, ndim)*augmentation
            augmentation = augmentation.repeat(len(x), 1).to(x.device)
            x = x.index_add(1, torch.arange(ndim).to(x.device), augmentation)
        
        edge_index = g.edge_index
        
        return self.propagate(
                edge_index, size=(x.size(0), x.size(0)),
                x=x)
                       
    def loss(self, g, loss_type= 'MAE', augment=True, augmentation=3, delta=1.0, **kwargs):
        if loss_type == 'MSE':
            return torch.sum((g.y - self.just_derivative(g, augment=augment, augmentation=augmentation))**2)
        if loss_type == 'MAE':
            return torch.sum(torch.abs(g.y - self.just_derivative(g, augment=augment, augmentation=augmentation)))
        if loss_type == 'HUBER':
            y_true = g.y
            y_pred = self.just_derivative(g, augment=augment, augmentation=augmentation)
            residual = torch.abs(y_true - y_pred)
            condition = residual < delta
            squared_loss = 0.5 * torch.pow(residual, 2)
            linear_loss = delta * (residual - 0.5 * delta)
            loss = torch.where(condition, squared_loss, linear_loss)
            return torch.sum(loss)


class varGN(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, hidden=300, aggr='add'):
        super(varGN, self).__init__(aggr=aggr)  # "Add" aggregation.
        self.msg_fnc = Seq(
            Lin(2*n_f, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
#             Lin(hidden, hidden),
#             ReLU(),
            Lin(hidden, msg_dim*2) #mu, logvar
        )
        
        self.node_fnc = Seq(
            Lin(msg_dim+n_f, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
#             Lin(hidden, hidden),
#             ReLU(),
            Lin(hidden, ndim)
        )
        self.sample = True
    
    #[docs]
    def forward(self, x, edge_index):
        #x is [n, n_f]
        x = x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
      
    def message(self, x_i, x_j):
        # x_i has shape [n_e, n_f]; x_j has shape [n_e, n_f]
        tmp = torch.cat([x_i, x_j], dim=1)  # tmp has shape [E, 2 * in_channels]
        raw_msg = self.msg_fnc(tmp)
        mu = raw_msg[:, 0::2]
        logvar = raw_msg[:, 1::2]
        actual_msg = mu
        if self.sample:
            actual_msg += torch.randn(mu.shape).to(x_i.device)*torch.exp(logvar/2)

        return actual_msg
    
    def update(self, aggr_out, x=None):
        # aggr_out has shape [n, msg_dim]

        tmp = torch.cat([x, aggr_out], dim=1)
        return self.node_fnc(tmp) #[n, nupdate]


class varOGN(varGN):
    def __init__(
		self, n_f, msg_dim, ndim, dt,
		edge_index, aggr='add', hidden=300, nt=1):

        super(varOGN, self).__init__(n_f, msg_dim, ndim, hidden=hidden, aggr=aggr)
        self.dt = dt
        self.nt = nt
        self.edge_index = edge_index
        self.ndim = ndim
    
    def just_derivative(self, g, augment=False, augmentation=3):
        #x is [n, n_f]f
        x = g.x
        ndim = self.ndim
        if augment:
            augmentation = torch.randn(1, ndim)*3
            augmentation = augmentation.repeat(len(x), 1).to(x.device)
            x = x.index_add(1, torch.arange(ndim).to(x.device), augmentation)
        
        edge_index = g.edge_index
        
        return self.propagate(
                edge_index, size=(x.size(0), x.size(0)),
                x=x)
    
    def loss(self, g, loss_type= 'MAE', augment=True, augmentation=3, delta=1.0, **kwargs):
        if loss_type == 'MSE':
            return torch.sum((g.y - self.just_derivative(g, augment=augment, augmentation=augmentation))**2)
        if loss_type == 'MAE':
            return torch.sum(torch.abs(g.y - self.just_derivative(g, augment=augment, augmentation=augmentation)))
        if loss_type == 'HUBER':
            y_true = g.y
            y_pred = self.just_derivative(g, augment=augment, augmentation=augmentation)
            residual = torch.abs(y_true - y_pred)
            condition = residual < delta
            squared_loss = 0.5 * torch.pow(residual, 2)
            linear_loss = delta * (residual - 0.5 * delta)
            loss = torch.where(condition, squared_loss, linear_loss)
            return torch.sum(loss)


#Personalized Models:
###################################################################################################################################################################

class GN_plusminus(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, hidden=300, aggr='add'):
        super(GN_plusminus, self).__init__(aggr=aggr)# "Add" aggregation.

        self.msg_fnc = Seq(
      Lin(2*n_f, hidden),
      ReLU(),
      Lin(hidden, int(hidden*3/2)),
      ReLU(),
      Lin(int(hidden*3/2),hidden*2),
      ReLU(),
      Lin(hidden*2, int(hidden*3/2)),
      ReLU(),
      Lin(int(hidden*3/2), hidden),
      ReLU(),
      Lin(hidden, int(hidden/3))
    )
  
        self.node_fnc = Seq(
      Lin(msg_dim+n_f, hidden),
      ReLU(),
      Lin(hidden, hidden),
      ReLU(),
      Lin(hidden, hidden),
      ReLU(),
      Lin(hidden, ndim)
      )
    
    def forward(self, x, edge_index):
        #x is [n, n_f]
        x = x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
    
    def message(self, x_i, x_j):
        # x_i has shape [n_e, n_f]; x_j has shape [n_e, n_f]
        tmp = torch.cat([x_i, x_j], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.msg_fnc(tmp)
    
    def update(self, aggr_out, x=None):
        # aggr_out has shape [n, msg_dim]

        tmp = torch.cat([x, aggr_out], dim=1)
        return self.node_fnc(tmp) #[n, nupdate]

class PM_GN(GN_plusminus):
    def __init__(self, n_f, msg_dim, ndim, dt,
        edge_index, aggr='add', hidden=300, nt=1):

        super(PM_GN, self).__init__(n_f, msg_dim, ndim, hidden=hidden, aggr=aggr)
        self.dt = dt
        self.nt = nt
        self.edge_index = edge_index
        self.ndim = ndim

    def just_derivative(self, g, augment=False, augmentation=3):
        #x is [n, n_f]f
        x = g.x
        ndim = self.ndim
        if augment:
            augmentation = torch.randn(1, ndim)*augmentation
            augmentation = augmentation.repeat(len(x), 1).to(x.device)
            x = x.index_add(1, torch.arange(ndim).to(x.device), augmentation)
        
        edge_index = g.edge_index
        
        return self.propagate(
                edge_index, size=(x.size(0), x.size(0)),
                x=x)

                       
    def loss(self, g, loss_type= 'MAE', augment=True, augmentation=3, delta=1.0, **kwargs):
        if loss_type == 'MSE':
            return torch.sum((g.y - self.just_derivative(g, augment=augment, augmentation=augmentation))**2)
        if loss_type == 'MAE':
            return torch.sum(torch.abs(g.y - self.just_derivative(g, augment=augment, augmentation=augmentation)))
        if loss_type == 'HUBER':
            y_true = g.y
            y_pred = self.just_derivative(g, augment=augment, augmentation=augmentation)
            residual = torch.abs(y_true - y_pred)
            condition = residual < delta
            squared_loss = 0.5 * torch.pow(residual, 2)
            linear_loss = delta * (residual - 0.5 * delta)
            loss = torch.where(condition, squared_loss, linear_loss)
            return torch.sum(loss)



class Custom_GN(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, hidden=300, aggr='add'):
        super(Custom_GN, self).__init__(aggr=aggr)  # "Add" aggregation.
        # self.msg_fnc = Seq(
        #     Lin(2*n_f, hidden),
        #     ReLU(),
        #     Lin(hidden, hidden),
        #     ReLU(),
        #     Lin(hidden, hidden),
        #     ReLU(),
        #     Lin(hidden, msg_dim)
        # )

        self.node_fnc = Seq(
            Lin(msg_dim+n_f, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, ndim)
        )
        self.msg_input_lin = Lin(2*n_f, hidden)
        self.msg_inverse = Lin(hidden, hidden)
        self.msg_inverse_quad = Lin(hidden, hidden)
        self.msg_out = Lin(2*hidden, msg_dim)
        self.activation = ReLU()


    def msg_fnc(self,x):
        x = self.msg_input_lin(x)
        x = self.activation(x)
        inv = self.msg_inverse(1/(x+1))
        inv_quad = self.msg_inverse_quad(1/(x+1)**2)
        concat = torch.cat([inv, inv_quad], dim=1)
        out = self.msg_out(concat)
        return out
    
    def forward(self, x, edge_index):
          #x is [n, n_f]
          x = x
          return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j):
        # x_i has shape [n_e, n_f]; x_j has shape [n_e, n_f]
        tmp = torch.cat([x_i, x_j], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.msg_fnc(tmp)

    def update(self, aggr_out, x=None):
        # aggr_out has shape [n, msg_dim]

        tmp = torch.cat([x, aggr_out], dim=1)
        return self.node_fnc(tmp) #[n, nupdate]


class CUST_GN(Custom_GN):
    def __init__(self, n_f, msg_dim, ndim, dt,
        edge_index, aggr='add', hidden=300, nt=1):

        super(CUST_GN, self).__init__(n_f, msg_dim, ndim, hidden=hidden, aggr=aggr)
        self.dt = dt
        self.nt = nt
        self.edge_index = edge_index
        self.ndim = ndim

    def just_derivative(self, g, augment=False, augmentation=3):
        #x is [n, n_f]f
        x = g.x
        ndim = self.ndim
        if augment:
            augmentation = torch.randn(1, ndim)*augmentation
            augmentation = augmentation.repeat(len(x), 1).to(x.device)
            x = x.index_add(1, torch.arange(ndim).to(x.device), augmentation)
        
        edge_index = g.edge_index
        
        return self.propagate(
                edge_index, size=(x.size(0), x.size(0)),
                x=x)

                       
    def loss(self, g, loss_type= 'MAE', augment=True, augmentation=3, delta=1.0, **kwargs):
        if loss_type == 'MSE':
            return torch.sum((g.y - self.just_derivative(g, augment=augment, augmentation=augmentation))**2)
        if loss_type == 'MAE':
            return torch.sum(torch.abs(g.y - self.just_derivative(g, augment=augment, augmentation=augmentation)))
        if loss_type == 'HUBER':
            y_true = g.y
            y_pred = self.just_derivative(g, augment=augment, augmentation=augmentation)
            residual = torch.abs(y_true - y_pred)
            condition = residual < delta
            squared_loss = 0.5 * torch.pow(residual, 2)
            linear_loss = delta * (residual - 0.5 * delta)
            loss = torch.where(condition, squared_loss, linear_loss)
            return torch.sum(loss)

