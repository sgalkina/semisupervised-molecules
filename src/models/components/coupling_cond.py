import torch
import torch.nn as nn
import torch.nn.functional as F
from models.components.basic import GraphLinear, GraphConv
import math


class AffineCoupling(nn.Module):  # delete
    def __init__(self, in_channel, hidden_channels, affine=True, mask_swap=False):  # filter_size=512,  --> hidden_channels =(512, 512)
        super(AffineCoupling, self).__init__()

        self.affine = affine
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.mask_swap=mask_swap
        # self.norms_in = nn.ModuleList()
        last_h = math.ceil(in_channel // 2)
        # print(in_channel, 'in_channel')
        # last_h = math.ceil(68)
        if affine:
            vh = tuple(hidden_channels) + (in_channel,)
        else:
#             vh = tuple(hidden_channels) + (in_channel // 2,)
            vh = tuple(hidden_channels) + (math.ceil(in_channel / 2),)

        for h in vh:
            self.layers.append(nn.Conv2d(last_h, h, kernel_size=3, padding=1))
            self.norms.append(nn.BatchNorm2d(h))  # , momentum=0.9 may change norm later, where to use norm? for the residual? or the sum
            # self.norms.append(ActNorm(in_channel=h, logdet=False)) # similar but not good
            last_h = h

       # Convolutional branch for the first tensor
        self.conv_branch = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3, padding=1),  # assuming input channels=9
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        # Calculate the output size after the conv layers
        # In this case, the output will be flattened to (batch_size, 32*1*1) due to pooling
        
        # Fully connected branch for the second tensor
        self.fc_branch = nn.Sequential(
            nn.Linear(6346, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        # Combined fully connected layers
        self.combined_fc = nn.Sequential(
            nn.Linear(32 + 128, 128),  # input size is the sum of the two branches
            nn.ReLU(),
            nn.BatchNorm1d(128),
            # nn.Linear(128, in_channel * 3 * 3),  # QM9
            nn.Linear(128, in_channel * 2 * 2),  # HMDB
            nn.ReLU()
        )
        
        # Final layer to reshape the output to (batch_size, 9, 3, 3)
        self.output_layer = nn.Sequential(
            # nn.Unflatten(1, (in_channel, 3, 3))  # QM9
            nn.Unflatten(1, (in_channel, 2, 2))  # HMDB
        )

    def forward(self, input, C):
        in_a, in_b = input.chunk(2, 1)  # (2,12,32,32) --> (2,6,32,32), (2,6,32,32)
        if self.mask_swap:
            in_a, in_b = in_b, in_a

        if self.affine:
            # log_s, t = self.net(in_a).chunk(2, 1)  # (2,12,32,32) --> (2,6,32,32), (2,6,32,32)
            s, t = self._s_t_function(in_a, C)
            out_b = (in_b + t) * s   #  different affine bias , no difference to the log-det # (2,6,32,32) More stable, less error
            # out_b = in_b * s + t
            logdet = torch.sum(torch.log(torch.abs(s)).view(input.shape[0], -1), 1)
        else:  # add coupling
            # net_out = self.net(in_a)
            _, t = self._s_t_function(in_a, C)
            out_b = in_b + t
            logdet = None

        if self.mask_swap:
            result = torch.cat([out_b, in_a], 1)
        else:
            result = torch.cat([in_a, out_b], 1)

        return result, logdet

    def reverse(self, output, C):
        out_a, out_b = output.chunk(2, 1)
        if self.mask_swap:
            out_a, out_b = out_b, out_a

        if self.affine:
            s, t = self._s_t_function(out_a, C)
            in_b = out_b / s - t  # More stable, less error   s must not equal to 0!!!
            # in_b = (out_b - t) / s
        else:
            _, t = self._s_t_function(out_a, C)
            in_b = out_b - t

        if self.mask_swap:
            result = torch.cat([in_b, out_a], 1)
        else:
            result = torch.cat([out_a, in_b], 1)

        return result

    def _s_t_function(self, x, C):
        h = x
        # print(x.shape, C.shape)
        # h = torch.cat([x, C], 1)
        # context_reshape = C.view(x.shape[0], -1, x.shape[2], x.shape[-1])
        # h = torch.cat([x, context_reshape], 1)
        for i in range(len(self.layers)-1):
            h = self.layers[i](h)
            h = self.norms[i](h)
            # h = torch.tanh(h)  # tanh may be more stable?
            h = torch.relu(h)  #
        h = self.layers[-1](h)
#         print('h shape is ' + str(h.shape))

        # Process the first tensor through the convolutional branch
        conv_out = self.conv_branch(h)
        # Process the second tensor through the fully connected branch
        fc_out = self.fc_branch(C)
        # Concatenate the outputs from both branches
        combined = torch.cat((conv_out, fc_out), dim=1)
        # Pass the concatenated features through the combined fully connected layers
        fc_out = self.combined_fc(combined)
        # Reshape the output to (batch_size, 9, 3, 3)
        h = self.output_layer(fc_out)
        # print(h.shape)

        s = None
        if self.affine:
            # residual net for doubling the channel. Do not use residual, unstable
            log_s, t = h.chunk(2, 1)
#             print('for adj, log_s is {}, t is {}'.format(log_s.shape, t.shape))
            # s = torch.sigmoid(log_s + 2)  # (2,6,32,32) # s != 0 and t can be arbitrary : Why + 2??? more stable, keep s != 0!!! exp is not stable
            s = torch.sigmoid(log_s)  # works good when actnorm
            # s = torch.tanh(log_s) # can not use tanh
            # s = torch.sign(log_s) # lower reverse error if no actnorm, similar results when have actnorm
        else:
            t = h
        return s, t


class GraphAffineCoupling(nn.Module):
    def __init__(self, n_node, in_dim, hidden_dim_dict, masked_row, affine=True):
        super(GraphAffineCoupling, self).__init__()
        self.n_node = n_node
        self.in_dim = in_dim
        self.hidden_dim_dict = hidden_dim_dict
        self.masked_row = masked_row
        self.affine = affine

        self.hidden_dim_gnn = hidden_dim_dict['gnn']
        self.hidden_dim_linear = hidden_dim_dict['linear']

        self.net = nn.ModuleList()
        self.norm = nn.ModuleList()
        last_dim = in_dim
        for out_dim in self.hidden_dim_gnn:  # What if use only one gnn???
            self.net.append(GraphConv(last_dim, out_dim))
            self.norm.append(nn.BatchNorm1d(n_node))  # , momentum=0.9 Change norm!!!
            # self.norm.append(ActNorm2D(in_dim=n_node, logdet=False))
            last_dim = out_dim

        self.net_lin = nn.ModuleList()
        self.norm_lin = nn.ModuleList()
        for i, out_dim in enumerate(self.hidden_dim_linear):  # What if use only one gnn???
            self.net_lin.append(GraphLinear(last_dim, out_dim))
            self.norm_lin.append(nn.BatchNorm1d(n_node))  # , momentum=0.9 Change norm!!!
            # self.norm_lin.append(ActNorm2D(in_dim=n_node, logdet=False))
            last_dim = out_dim

        if affine:
            self.net_lin.append(GraphLinear(last_dim, in_dim*2))
        else:
            self.net_lin.append(GraphLinear(last_dim, in_dim))

        self.scale = nn.Parameter(torch.zeros(1))  # nn.Parameter(torch.ones(1)) #
        mask = torch.ones(n_node, in_dim)
        mask[masked_row, :] = 0  # masked_row are kept same, and used for _s_t for updating the left rows
        self.register_buffer('mask', mask)

        # Process the first tensor (shaped (batch_size, 9, 5))
        self.conv_branch = nn.Sequential(
            nn.Conv1d(in_channels=9, out_channels=32, kernel_size=3, padding=1),  # QM9
            # nn.Conv1d(in_channels=38, out_channels=32, kernel_size=3, padding=1),  # HMDB
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )
        # Calculate the output size after the conv layers
        # Assuming the pooling reduces the size to 2, the output will be flattened to (batch_size, 32*2)
        
        # Process the second tensor (shaped (batch_size, 200))
        self.fc_branch = nn.Sequential(
            nn.Linear(6346, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        # Combined fully connected layers
        self.combined_fc = nn.Sequential(
            nn.Linear(288, 256),  # QM9
            # nn.Linear(608, 256),  # HMDB
            nn.ReLU(),
            nn.BatchNorm1d(256),
            # nn.Linear(256, 38*30),  # HMDB
            nn.Linear(256, 9*10),  # QM9
            nn.ReLU()
        )
        
        # Final layer to reshape the output to (batch_size, 9, 5)
        self.output_layer = nn.Sequential(
            # nn.Unflatten(1, (9, 10)) # QM9
            nn.Unflatten(1, (38, 30)) # HMDB
        )

    def forward(self, adj, input, C):
#         print('input shape is '+ str(input.shape))
        masked_x = self.mask * input
#         print('masked_X shape is '+ str(masked_x.shape))
        s, t = self._s_t_function(adj, masked_x, C)  # s must not equal to 0!!!
        if self.affine:
            out = masked_x + (1-self.mask) * (input + t) * s
            # out = masked_x + (1-self.mask) * (input * s + t)
            logdet = torch.sum(torch.log(torch.abs(s)).view(input.shape[0], -1), 1)  # possibly wrong answer
        else:  # add coupling
            out = masked_x + t*(1-self.mask)
            logdet = None
        return out, logdet

    def reverse(self, adj, output, C):
        masked_y = self.mask * output
        s, t = self._s_t_function(adj, masked_y, C)
        if self.affine:
            input = masked_y + (1 - self.mask) * (output / s - t)
            # input = masked_x + (1 - self.mask) * ((output-t) / s)
        else:
            input = masked_y + (1 - self.mask) * (output - t)
        return input

    def _s_t_function(self, adj, x, C):
        # adj: (2,4,9,9)  x: # (2,9,5)
        s = None
        h = x
        # context_reshape = C.view(h.shape[0], h.shape[1], -1)
        # h = torch.cat([h, context_reshape], 2)
        for i in range(len(self.net)):
            h = self.net[i](adj, h)  # (2,1,9,hidden_dim)
            h = self.norm[i](h)
            # h = torch.tanh(h)  # tanh may be more stable
            h = torch.relu(h)  # use relu!!!

        for i in range(len(self.net_lin)-1):
            h = self.net_lin[i](h)  # (2,1,9,hidden_dim)
            h = self.norm_lin[i](h)
            # h = torch.tanh(h)
            h = torch.relu(h)

        h = self.net_lin[-1](h)
        # h =h * torch.exp(self.scale*2)

        # Process the first tensor through the convolutional branch
        conv_out = self.conv_branch(h)
        # Process the second tensor through the fully connected branch
        fc_out = self.fc_branch(C)
        # Concatenate the outputs from both branches
        combined = torch.cat((conv_out, fc_out), dim=1)
        # Pass the concatenated features through the combined fully connected layers
        fc_out = self.combined_fc(combined)
        # Reshape the output to (batch_size, 9, 3, 3)
        h = self.output_layer(fc_out)

        if self.affine:
            log_s, t = h.chunk(2, dim=-1)
#             print('for node, log_s is {}, t is {}'.format(log_s.shape, t.shape))
            #  x = sigmoid(log_x+bias): glow code Top 1 choice, keep s away from 0, s!!!!= 0  always safe!!!
            # And get the signal from added noise in the  input
            # s = torch.sigmoid(log_s + 2)
            s = torch.sigmoid(log_s)  # better validity + actnorm

            # s = torch.tanh(log_s)  # Not stable when s =0 for synthesis data, but works well for real data in best case....
            # s = torch.sign(s)

            # s = torch.sign(log_s)

            # s = F.softplus(log_s) # negative nll
            # s = torch.sigmoid(log_s)  # little worse than +2, # *self.scale #!!! # scale leads to nan results
            # s = torch.tanh(log_s+2) # not that good
            # s = torch.relu(log_s) # nan results
            # s = log_s  # nan results
            # s = torch.exp(log_s)  # nan results
        else:
            t = h
        return s, t
