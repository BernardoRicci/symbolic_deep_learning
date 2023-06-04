import numpy as np
import torch
from torch import nn
from torch.functional import F
from torch.optim import Adam
from torch_geometric.nn import MetaLayer, MessagePassing, GATConv
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
    
    def huber_loss(prediction, target, delta):
	    absolute_difference = torch.abs(prediction - target)
	    quadratic_term = 0.5 * (absolute_difference ** 2)
	    linear_term = delta * (absolute_difference - 0.5 * delta)
	    loss = torch.where(absolute_difference <= delta, quadratic_term, linear_term)
	    return torch.mean(loss)

                       
    def loss(self, g, loss_type= 'mae'):
        if loss_type == 'mse':
            return torch.sum((g.y - self.just_derivative(g, augment=augment, augmentation=augmentation))**2)
        if loss_type == 'mae':
            return torch.sum(torch.abs(g.y - self.just_derivative(g, augment=augment, augmentation=augmentation)))
        if loss_type == 'huber':
            return huber_loss(g.y, self.just_derivative(g, augment=augment, augmentation=augmentation), delta)


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

                       
    def huber_loss(prediction, target, delta):
	    absolute_difference = torch.abs(prediction - target)
	    quadratic_term = 0.5 * (absolute_difference ** 2)
	    linear_term = delta * (absolute_difference - 0.5 * delta)
	    loss = torch.where(absolute_difference <= delta, quadratic_term, linear_term)
	    return torch.mean(loss)

                       
    def loss(self, g, loss_type= 'mae'):
        if loss_type == 'mse':
            return torch.sum((g.y - self.just_derivative(g, augment=augment, augmentation=augmentation))**2)
        if loss_type == 'mae':
            return torch.sum(torch.abs(g.y - self.just_derivative(g, augment=augment, augmentation=augmentation)))
        if loss_type == 'huber':
            return huber_loss(g.y, self.just_derivative(g, augment=augment, augmentation=augmentation), delta)



class GATLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, num_heads=1):
        super(GATLayer, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        self.linear = nn.Linear(in_channels, out_channels * num_heads, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, num_heads, 2 * out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index):
        x = self.linear(x).view(-1, self.num_heads, self.out_channels)

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, edge_index_i, size_i):
        alpha = torch.cat([x_i, x_j], dim=-1)
        alpha = (alpha * self.att).sum(dim=-1)
        alpha = softmax(alpha, edge_index_i, size_i)

        return x_j * alpha.view(-1, self.num_heads, 1)

    def update(self, aggr_out):
        return aggr_out.view(-1, self.out_channels * self.num_heads)

class GAT_GN(nn.Module):
	def __init__(self, n_f, edge_index, hidden=300, out_dim=100, num_heads=4, num_layers=3):
		super(GAT_GN, self).__init__()
		self.num_layers = num_layers
		self.layers = nn.ModuleList()
		self.edge_index = edge_index

		self.layers.append(GATLayer(n_f, hidden, num_heads))

		for _ in range(num_layers - 2):
			self.layers.append(GATLayer(hidden * num_heads, hidden, num_heads))

		self.layers.append(GATLayer(hidden * num_heads, out_dim, 1))

	def just_derivative(self, g, augment=False, augmentation=3):
		#x is [n, n_f]f
		x = g.x
		ndim = self.ndim
		if augment:
			augmentation = torch.randn(1, ndim)*augmentation
			augmentation = augmentation.repeat(len(x), 1).to(x.device)
			x = x.index_add(1, torch.arange(ndim).to(x.device), augmentation)

		edge_index = g.edge_index

		return self.propagate(edge_index, size=(x.size(0), x.size(0)),x=x)


def huber_loss(prediction, target, delta):
	absolute_difference = torch.abs(prediction - target)
	quadratic_term = 0.5 * (absolute_difference ** 2)
	linear_term = delta * (absolute_difference - 0.5 * delta)
	loss = torch.where(absolute_difference <= delta, quadratic_term, linear_term)
	return torch.mean(loss)


def loss(self, g, loss_type= 'mae'):
	if loss_type == 'mse':
		return torch.sum((g.y - self.just_derivative(g, augment=augment, augmentation=augmentation))**2)
	if loss_type == 'mae':
		return torch.sum(torch.abs(g.y - self.just_derivative(g, augment=augment, augmentation=augmentation)))
	if loss_type == 'huber':
		return huber_loss(g.y, self.just_derivative(g, augment=augment, augmentation=augmentation), delta)
