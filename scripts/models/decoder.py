import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F
from pytorch3d import transforms
from torch.nn import Parameter


class MaskDecoder(nn.Module):
    def __init__(self, latent_dim=256, img_channels=1, img_size=512):
        super(MaskDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size
        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # (batch_size, 256, 16, 16)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (batch_size, 128, 32, 32)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # (batch_size, 64, 64, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # (batch_size, 32, 128, 128)
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # (batch_size, 16, 256, 256)
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, img_channels, kernel_size=4, stride=2, padding=1), # (batch_size, 1, 512, 512)
            nn.Tanh() )

        

    def forward(self, z):
        batch_size = z.size(0)
        x = self.fc(z)
        x = x.view(batch_size, 512, 8, 8)
        img = self.decoder(x)
        return img


class SDFDecoder(nn.Module):
    def __init__(self, dropout_prob=0.2):
        super(SDFDecoder, self).__init__()
        self.fc_stack_1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(258, 512)),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.utils.weight_norm(nn.Linear(512, 254)),  # 510 = 512 - 2
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        self.fc_stack_2 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.utils.weight_norm(nn.Linear(512, 1))
        )
        self.th = nn.Tanh()

    def forward(self, x):
        skip_out = self.fc_stack_1(x)
        skip_in = torch.cat([skip_out, x], 2)
        y = self.fc_stack_2(skip_in)
        out = self.th(y)
        return out


class UDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 scale=1,
                 bias=0.5,
                d_in_spatial =2,
                 geometric_init=True,
                 weight_norm=True,
                 udf_type='abs',
                 ):
        super(UDFNetwork, self).__init__()
        self.lat_dim = d_in
        d_in = d_in + d_in_spatial
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        
        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch
        # self.mapping = MappingNet(self.lat_dim, self.lat_dim)
        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        self.geometric_init = geometric_init

        # self.bias = 0.5
        # bias = self.bias
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                print("using geometric init")
                if l == self.num_layers - 2:

                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)

                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)
        self.relu = nn.ReLU()
        self.udf_type = udf_type

    def udf_out(self, x):
        if self.udf_type == 'abs':
            return torch.abs(x)
        elif self.udf_type == 'square':
            return x ** 2
        elif self.udf_type == 'sdf':
            return x

    def forward(self, inputs):
       # lat_rep = self.mapping(lat_rep)
        # inputs = xyz * self.scale
        # inputs = torch.cat([inputs, lat_rep], dim=-1)
        inputs = inputs
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.concat([x, inputs], dim=-1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return self.udf_out(x)
        # return torch.cat([self.udf_out(x[:, :1]) / self.scale, x[:, 1:]],
        #                  dim=-1)

    def udf(self, xyz, latent):
        # Concatenate xyz and latent to match forward signature
        inputs = torch.cat([xyz, latent], dim=-1)
        return self.forward(inputs)

    def udf_hidden_appearance(self, xyz, latent):
        return self.forward(xyz, latent)

    def gradient(self, xyz, latent):
        d_output = torch.ones_like(latent, requires_grad=False, device=xyz.device)
        gradients = torch.autograd.grad(
            outputs=xyz,
            inputs=latent,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

class BoneDecoder(nn.Module):
    def __init__(self, num_bones, latent_dim=256):
        super(BoneDecoder, self).__init__()
        self.num_bones = num_bones
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(latent_dim, 256)  
        self.fc2 = nn.Linear(256, 256)         
        self.fc_bone = nn.Linear(256, num_bones * 3)  
        
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_bone(x)
        x = x.view(-1, self.num_bones, 3)
        centers = x[:, :, :3]
        centers = torch.clamp(centers, min=-1, max=1)
        return centers.squeeze(0)

class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs=10, logscale=True, alpha=None):
        """
        adapted from https://github.com/kwea123/nerf_pl/blob/master/models/nerf.py
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.nfuncs = len(self.funcs)
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)
        if alpha is None:
            self.alpha = self.N_freqs
        else: self.alpha = alpha

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        # consine features
        if self.N_freqs>0:
            shape = x.shape
            bs = shape[0]
            input_dim = shape[-1]
            output_dim = input_dim*(1+self.N_freqs*self.nfuncs)
            out_shape = shape[:-1] + ((output_dim),)
            device = x.device

            x = x.view(-1,input_dim)
            out = []
            for freq in self.freq_bands:
                for func in self.funcs:
                    out += [func(freq*x)]
            out =  torch.cat(out, -1)

            ## Apply the window w = 0.5*( 1+cos(pi + pi clip(alpha-j)) )
            out = out.view(-1, self.N_freqs, self.nfuncs, input_dim)
            window = self.alpha - torch.arange(self.N_freqs).to(device)
            window = torch.clamp(window, 0.0, 1.0)
            window = 0.5 * (1 + torch.cos(np.pi * window + np.pi))
            window = window.view(1,-1, 1, 1)
            out = window * out
            out = out.view(-1,self.N_freqs*self.nfuncs*input_dim)

            out = torch.cat([x, out],-1)
            out = out.view(out_shape)
        else: out = x
        return out

class TransPredictor(nn.Module):
    def __init__(self, num_bones, latent_dim=128, center_dim=3, use_quat=True, D=8, W=256):
        super(TransPredictor, self).__init__() 
        self.num_bones = num_bones
        self.latent_dim = latent_dim
        self.center_dim = center_dim
        self.use_quat = use_quat
        self.D = D
        self.W = W
        self.pe = Embedding(3, N_freqs=10, logscale=True, alpha=10)

        input_dim = latent_dim + 63

        self.layers = nn.ModuleList()
        for i in range(D):
            in_features = input_dim if i == 0 else W
            out_features = W
            self.layers.append(nn.Sequential(
                nn.Linear(in_features, out_features),
               nn.ReLU(inplace=True),
            ))
            input_dim = W  

        if self.use_quat:
            self.rotation_output = nn.Linear(W,   7)  # quaternion output
        else:
            self.rotation_output = nn.Linear(W, 6)  # rotation output

    def forward(self, latent_code, centers,return_rot=True):
        x = self.pe(centers)    
        x= x.unsqueeze(0).repeat(latent_code.shape[0],1,1)
        latent_code_expanded = latent_code.unsqueeze(1).expand(-1, x.shape[1], -1)        
        x = torch.cat([x, latent_code_expanded], dim=2)  # [b, B, 63+lat_dim]
        # x = x.squeeze()
        for layer in self.layers:
            x = layer(x)
        
        rotations = self.rotation_output(x)
        tmat = rotations[:,:, :3]
        if self.use_quat:
            rquat = rotations[:,:,3:7]
            rquat= F.normalize(rquat, p=2, dim=-1)
            rmat = transforms.quaternion_to_matrix(rquat)
        else:
            rot  = x[:, 3:6]
            rmat = transforms.so3_exponential_map(rot)
            
        rmat = rmat.view(rmat.shape[0],rmat.shape[1], -1)
        rts = torch.cat([rmat, tmat], -1)
        # rts = rts.view(x.shape[0],1,-1)
        if  return_rot:
            return rotations
        else:
            return rts


class SWPredictor(nn.Module):
    def __init__(self, num_bones, latent_dim=256, center_dim=3, use_quat=True, D=8, W=256):
        super(SWPredictor, self).__init__() 
        self.num_bones = num_bones
        self.latent_dim = latent_dim
        self.center_dim = center_dim
        self.use_quat = use_quat
        self.D = D
        self.W = W
        self.pe = Embedding(3, N_freqs=10, logscale=True, alpha=10)

        input_dim = latent_dim + 63

        self.layers = nn.ModuleList()
        for i in range(D):
            in_features = input_dim if i == 0 else W
            out_features = W
            self.layers.append(nn.Sequential(
                nn.Linear(in_features, out_features),nn.ReLU(inplace=True),
            ))
            input_dim = W  
        self.bone_head = nn.Linear(W, num_bones)


    def forward(self, latent_code, centers):
        x = self.pe(centers)
        latent_code_expanded = latent_code.unsqueeze(1).expand(-1, centers.size(1), -1)        
        x = torch.cat([x, latent_code_expanded], dim=2)  # [1, B, 63+lat_dim]
        # x = x.squeeze()
        for layer in self.layers:
            x = layer(x)
        sw = self.bone_head(x)
        sw = F.softmax(sw, dim=-1)
        return sw

def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim