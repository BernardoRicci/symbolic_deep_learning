import numpy as np
import torch
from torch import nn
from torch.functional import F
from torch.optim import Adam
from torch_geometric.nn import MetaLayer, MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Softplus
from torch.autograd import Variable, grad

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
    
    def loss(self, g, augment=True, square=False, augmentation=3, **kwargs):
        if square:
            return torch.sum((g.y - self.just_derivative(g, augment=augment, augmentation=augmentation))**2)
        else:
            return torch.sum(torch.abs(g.y - self.just_derivative(g, augment=augment)))


#modelli personalizzati:
###################################################################################################################################################################

class our_GN(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, hidden=550, aggr='add'):
        super(our_GN, self).__init__(aggr=aggr)  # "Add" aggregation.
        self.msg_fnc = Seq(
            Lin(2*n_f, hidden),
            ReLU(),
            Lin(hidden, int(hidden*2./3.)),
            ReLU(),
            Lin(int(hidden*2./3.), int(hidden/2.)),
            ReLU(),
            Lin(int(hidden/2.), msg_dim)
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


class Fiasco_GN(our_GN):
    def __init__(self, n_f, msg_dim, ndim, dt,
        edge_index, aggr='add', hidden=300, nt=1):

        super(Fiasco_GN, self).__init__(n_f, msg_dim, ndim, hidden=hidden, aggr=aggr)
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
                  
    def loss(self, g, loss_type= 'abs', perc = 0.05):
        if loss_type == 'square':
            return torch.sum((g.y - self.just_derivative(g))**2)
        if loss_type == 'abs':
            return torch.sum(torch.abs(g.y - self.just_derivative(g)))
        if loss_type == 'rad': 
            return torch.sum(torch.sqrt(torch.abs(g.y - self.just_derivative(g))+0.01))

    
class GN_mbuti(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, hidden=200, aggr='add'):
        super(GN_mbuti, self).__init__(aggr=aggr)# "Add" aggregation.

        self.msg_fnc = Seq(
      Lin(2*n_f, hidden),
      ReLU(),
      Lin(hidden, int(hidden*3./2.)),
      ReLU(),
      Lin(int(hidden*3./2.),2*hidden),
      ReLU(),
      Lin(2*hidden, int(hidden*3./2.)),
      ReLU(),
      Lin(int(hidden*3./2.), hidden),
      ReLU(),
      Lin(hidden, int(hidden/2.))
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

class Mbuti_GN(GN_mbuti):
    def __init__(self, n_f, msg_dim, ndim, dt,
        edge_index, aggr='add', hidden=200, nt=1):

        super(Mbuti_GN, self).__init__(n_f, msg_dim, ndim, hidden=hidden, aggr=aggr)
        self.dt = dt
        self.nt = nt
        self.edge_index = edge_index
        self.ndim = ndim

    def just_derivative(self, g):
      #x is [n, n_f]f
      x = g.x
      ndim = self.ndim
      edge_index = g.edge_index

      return self.propagate(edge_index, size=(x.size(0), x.size(0)),x=x)
                       
    def loss(self, g, loss_type= 'abs'):
        if loss_type == 'square':
            return torch.sum((g.y - self.just_derivative(g))**2)
        if loss_type == 'abs':
            return torch.sum(torch.abs(g.y - self.just_derivative(g)))
        if loss_type == 'rad': 
            return torch.sum(torch.sqrt(torch.abs(g.y - self.just_derivative(g))+0.01))
