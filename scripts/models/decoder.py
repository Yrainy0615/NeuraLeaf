import torch
import torch.nn as nn
import numpy as np 

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
        return self.forward(xyz, latent)

    def udf_hidden_appearance(self, xyz, latent):
        return self.forward(xyz, latent)

    def gradient(self, xyz, latent):
        xyz.requires_grad_(True)
        latent.requires_grad_(True)
        y = self.udf(xyz, latent)
        x = torch.cat([xyz, latent],dim=-1)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
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