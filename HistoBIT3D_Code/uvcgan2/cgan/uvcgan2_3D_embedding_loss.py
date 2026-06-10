# pylint: disable=not-callable
# NOTE: Mistaken lint:
# E1102: self.criterion_gan is not callable (not-callable)

import itertools
import torch
import os
import matplotlib.pyplot as plt
import numpy as np


import torchvision.transforms.functional as TF
from torchvision.transforms import GaussianBlur, Resize
import torch.nn.functional as F

from uvcgan2.torch.select            import (
    select_optimizer, extract_name_kwargs
)
from uvcgan2.torch.queue             import FastQueue
from uvcgan2.torch.funcs             import prepare_model, update_average_model
from uvcgan2.torch.layers.batch_head import BatchHeadWrapper, get_batch_head
from uvcgan2.base.losses             import GANLoss
from uvcgan2.torch.gradient_penalty  import GradientPenalty
from uvcgan2.models.discriminator    import construct_discriminator
from uvcgan2.models.generator        import construct_generator

from .model_base import ModelBase
from .named_dict import NamedDict
from .funcs import set_two_domain_input, save_image, save_embedding_as_image

def construct_consistency_model(consist, device):
    name, kwargs = extract_name_kwargs(consist)

    if name == 'blur':
        return GaussianBlur(**kwargs).to(device)

    if name == 'resize':
        return Resize(**kwargs).to(device)

    raise ValueError(f'Unknown consistency type: {name}')

def queued_forward(batch_head_model, input_image, queue, update_queue = True):
    output, pred_body = batch_head_model.forward(
        input_image, extra_bodies = queue.query(), return_body = True
    )

    if update_queue:
        queue.push(pred_body)

    return output

class UVCGAN2_3D_embedding_loss(ModelBase):
    # pylint: disable=too-many-instance-attributes

    def _setup_images(self, _config):
        images = [
            'real_a', 'real_b',
            'fake_a', 'fake_b',
            'reco_a', 'reco_b',
            'consist_real_a', 'consist_real_b',
            'consist_fake_a', 'consist_fake_b',
            'real_a_z1', 'real_a_z2',
            'fake_b_z1', 'fake_b_z2',
            'real_a_name', 'real_a_adj_name',
        ]

        if self.is_train and self.lambda_idt > 0:
            images += [ 'idt_a', 'idt_b', ]

        return NamedDict(*images)

    def _construct_batch_head_disc(self, model_config, input_shape):
        disc_body = construct_discriminator(
            model_config, input_shape, self.device
        )

        disc_head = get_batch_head(self.head_config)
        disc_head = prepare_model(disc_head, self.device)

        return BatchHeadWrapper(disc_body, disc_head)

    def _setup_models(self, config):
        models = {}

        shape_a = config.data.datasets[0].shape
        shape_b = config.data.datasets[1].shape

        models['gen_ab'] = construct_generator(
            config.generator, shape_a, shape_b, self.device
        )
        models['gen_ba'] = construct_generator(
            config.generator, shape_b, shape_a, self.device
        )

        if self.avg_momentum is not None:
            models['avg_gen_ab'] = construct_generator(
                config.generator, shape_a, shape_b, self.device
            )
            models['avg_gen_ba'] = construct_generator(
                config.generator, shape_b, shape_a, self.device
            )

            models['avg_gen_ab'].load_state_dict(models['gen_ab'].state_dict())
            models['avg_gen_ba'].load_state_dict(models['gen_ba'].state_dict())

        if self.is_train:
            models['disc_a'] = self._construct_batch_head_disc(
                config.discriminator, config.data.datasets[0].shape
            )
            models['disc_b'] = self._construct_batch_head_disc(
                config.discriminator, config.data.datasets[1].shape
            )

        ## Register ViT Forward Hooks here for embedding loss: 
        embedding_storage = {}

        def make_hook(name):
            def hook(module, input, output):
                embedding_storage[name] = output.detach()
            return hook
        
        # Register hook to bottleneck (ExtendedPixelwiseViT) for both generators
        print("Modules inside gen_ab.net:")
        for name, module in models['gen_ab'].net.named_modules():
            print(name)

        # Get the bottleneck layers AB
        bottleneck_layer_ab = models['gen_ab'].net.modnet.inner_module.inner_module.inner_module.inner_module.encoder.encoder[11].norm2
        bottleneck_layer_ab.register_forward_hook(make_hook("ab"))
        
        bottleneck_layer_ba = models['gen_ba'].net.modnet.inner_module.inner_module.inner_module.inner_module.encoder.encoder[11].norm2
        bottleneck_layer_ba.register_forward_hook(make_hook("ba"))

        return NamedDict(**models)

    def _setup_losses(self, config):
        losses = [
            'gen_ab', 'gen_ba', 'cycle_a', 'cycle_b', 'disc_a', 'disc_b',
        ]

        if self.is_train and self.lambda_idt > 0:
            losses += [ 'idt_a', 'idt_b' ]

        if self.is_train and config.gradient_penalty is not None:
            losses += [ 'gp_a', 'gp_b' ]

        if self.consist_model is not None:
            losses += [ 'consist_a', 'consist_b' ]
        
        # This is the subtraction loss that we will start with. 
        if self.is_train and self.lambda_sub_loss > 0: 
            losses += [ 'subtraction_adj']
        
        # This is the embedding loss. 
        if self.is_train and self.lambda_embedding_loss > 0: 
            losses += [ 'embedding_adj']  

        return NamedDict(*losses)

    def _setup_optimizers(self, config):
        optimizers = NamedDict('gen', 'disc')

        optimizers.gen = select_optimizer(
            itertools.chain(
                self.models.gen_ab.parameters(),
                self.models.gen_ba.parameters()
            ),
            config.generator.optimizer
        )

        optimizers.disc = select_optimizer(
            itertools.chain(
                self.models.disc_a.parameters(),
                self.models.disc_b.parameters()
            ),
            config.discriminator.optimizer
        )

        return optimizers

    def __init__(
        self, savedir, config, is_train, device, head_config = None,
        lambda_a        = 10.0,
        lambda_b        = 10.0,
        lambda_idt      = 0.5,
        lambda_consist  = 0,
        lambda_subtraction_loss = 0.5, # This is the weight for the subtraction loss between z1 and z2 in the adjacent slice mode.
        lambda_embedding_loss = 0.5, # This is the weight for the embedding loss between adjacent slices.
        head_queue_size = 3,
        avg_momentum    = None,
        consistency     = None,
        z_spacing      = 1.0, # default. 
        debug_root =  None, # Optional root directory for saving debug images from the subtraction loss.
    ):
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        self.lambda_a       = lambda_a
        self.lambda_b       = lambda_b
        self.lambda_idt     = lambda_idt
        self.lambda_consist = lambda_consist
        self.lambda_sub_loss = lambda_subtraction_loss # subtraction Loss
        self.lambda_embedding_loss = lambda_embedding_loss # embedding Losss
        self.avg_momentum   = avg_momentum
        self.head_config    = head_config or {}
        self.consist_model  = None
        self.z_spacing      = z_spacing
        self.debug_root       = debug_root
        self.current_step =0

        print(f"Initialized UVCGAN2_3D_embedding_loss with lambda_a={lambda_a}, lambda_b={lambda_b}, lambda_idt={lambda_idt}, lambda_consist={lambda_consist}, lambda_subtraction_loss={lambda_subtraction_loss}, lambda_embedding_loss={lambda_embedding_loss}, avg_momentum={avg_momentum}, z_spacing={z_spacing}")

        # 🔥 Correct handling of debug_root
        if debug_root is not None:
            self.debug_root = debug_root
        elif hasattr(config, "debug_root"):
            self.debug_root = config.debug_root
        else:
            self.debug_root = None

        if self.debug_root is not None:
            os.makedirs(self.debug_root, exist_ok=True)

        if (lambda_consist > 0) and (consistency is not None):
            self.consist_model \
                = construct_consistency_model(consistency, device)

        assert len(config.data.datasets) == 2, \
            "CycleGAN expects a pair of datasets"

        super().__init__(savedir, config, is_train, device)

        self.criterion_gan     = GANLoss(config.loss).to(self.device)
        self.criterion_cycle   = torch.nn.L1Loss()
        self.criterion_idt     = torch.nn.L1Loss()
        self.criterion_consist = torch.nn.L1Loss()
        self.criterion_subtract = torch.nn.L1Loss()  # Loss for the subtraction between adjacent slices in domain A.
        self.criterion_embedding = torch.nn.L1Loss()  # Loss for the embedding consistency between adjacent slices in domain A.

        if self.is_train:
            self.queues = NamedDict(**{
                name : FastQueue(head_queue_size, device = device)
                    for name in [ 'real_a', 'real_b', 'fake_a', 'fake_b' ]
            })

            self.gp = None

            if config.gradient_penalty is not None:
                self.gp = GradientPenalty(**config.gradient_penalty)

    def _set_input(self, inputs, domain):
        set_two_domain_input(self.images, inputs, domain, self.device)

        # Debugging the shape and name
        # print(
        #     "DEBUG: real_a and real_a_adj loaded:",
        #     f"real_a shape = {None if self.images.real_a is None else tuple(self.images.real_a.shape)},",
        #     f"real_a_adj shape = {None if self.images.real_a_adj is None else tuple(self.images.real_a_adj.shape)}"
        # )

        # if hasattr(self.images, "real_a_names"):
        #     print("  real_a_names:", self.images.real_a_names)

        # if hasattr(self.images, "real_a_adj_names"):
        #     print("  real_a_adj_names:", self.images.real_a_adj_names)

        if self.images.real_a is not None:
            if self.consist_model is not None:
                self.images.consist_real_a \
                    = self.consist_model(self.images.real_a)

        if self.images.real_b is not None:
            if self.consist_model is not None:
                self.images.consist_real_b \
                    = self.consist_model(self.images.real_b)

    def cycle_forward_image(self, real, gen_fwd, gen_bkw):
        # pylint: disable=no-self-use

        # (N, C, H, W)
        fake = gen_fwd(real)
        reco = gen_bkw(fake)

        consist_fake = None

        if self.consist_model is not None:
            consist_fake = self.consist_model(fake)

        return (fake, reco, consist_fake)

    def embedding_loss(self, real_a, real_a_adj, gen_fwd, gen_bkw, step=None):
        """
        Compute motion consistency loss between real and generated slice pairs
        using ViT bottleneck embeddings.
        """

        # --------- 1. Setup embedding storage dict ---------
        if not hasattr(self, "embedding_storage"):
            self.embedding_storage = {}

        def make_hook(name):
            def hook(module, input, output):
                self.embedding_storage[name] = output.detach()
            return hook

        # --------- 2. Register forward hooks (only once) ---------
        if not hasattr(self, "hook_handles"):
            self.hook_handles = {}

            bottleneck_z = gen_fwd.net.modnet.inner_module.inner_module.inner_module.inner_module.encoder.encoder[11].norm2
            bottleneck_fake = gen_bkw.net.modnet.inner_module.inner_module.inner_module.inner_module.encoder.encoder[11].norm2

            self.hook_handles["z"] = bottleneck_z.register_forward_hook(make_hook("z"))
            self.hook_handles["fake"] = bottleneck_fake.register_forward_hook(make_hook("fake"))

        # --------- 3. Forward passes to trigger hooks ---------
        _ = gen_fwd(real_a)
        z1_vit = self.embedding_storage["z"].clone()

        _ = gen_fwd(real_a_adj)
        z2_vit = self.embedding_storage["z"].clone()

        fake_b = gen_fwd(real_a)
        fake_b_adj = gen_fwd(real_a_adj)

        _ = gen_bkw(fake_b)
        fake1_vit = self.embedding_storage["fake"].clone()

        _ = gen_bkw(fake_b_adj)
        fake2_vit = self.embedding_storage["fake"].clone()

        # --------- 4. Compute cosine-based motion maps ---------
        def cosine_motion(a, b):
            a, b = a[:-1], b[:-1]  # remove style token (last) # shape: (N, B, D), style token is the last token. Look at transformer.py to understand. it is concatenated as [patch_tokens, style_token]. We want to remove the style token for motion consistency.
            a = a.squeeze(1)     # (N, D)
            b = b.squeeze(1)

            cos_sim = F.cosine_similarity(a, b, dim=-1)  # (N,)
            motion = 1 - cos_sim                         # higher = more motion
            return motion

        motion_real = cosine_motion(z1_vit, z2_vit)
        motion_fake = cosine_motion(fake1_vit, fake2_vit)
        #print(f"DEBUG: motion_real shape = {motion_real.shape}, motion_fake shape = {motion_fake.shape}")

        # --------- 5. Debug output ---------
        if step is not None and step % 100 == 0:
            def shp(x): return None if x is None else tuple(x.shape)
            print(
                f"[ViT bottleneck @ step {step}] "
                f"z1={shp(z1_vit)}, "
                f"z2={shp(z2_vit)}, "
                f"fake1={shp(fake1_vit)}, "
                f"fake2={shp(fake2_vit)}"
            )

            save_image(real_a[0],          os.path.join(self.debug_root, "real_a_z.png"))
            save_image(real_a_adj[0],          os.path.join(self.debug_root, "real_a_adj_z.png"))
            save_image(fake_b[0],          os.path.join(self.debug_root, "fake_b_z.png"))
            save_image(fake_b_adj[0],          os.path.join(self.debug_root, "fake_b_adj_z.png"))
            save_embedding_as_image(z1_vit, self.debug_root, prefix="embedding_real")
            save_embedding_as_image(fake1_vit, self.debug_root, prefix="embedding_fake")
            save_embedding_as_image(motion_real, self.debug_root, prefix="motion_embedding_real")
            save_embedding_as_image(motion_fake, self.debug_root, prefix="motion_embedding_fake")


        # --------- 6. Motion consistency loss ---------
        return self.criterion_embedding(motion_real, motion_fake)


    # Visualize the ViT Bottle Neck Embeddings for debugging. 
    def save_embedding_as_image(tensor, save_path, prefix="embed", max_channels=3):
        """
        Saves selected channels or a channel-aggregated 2D projection of the embedding tensor.

        Args:
            tensor: (B, C, H, W) torch tensor
            save_path: folder to save the image(s)
            prefix: filename prefix
            max_channels: number of individual channels to save
        """
        os.makedirs(save_path, exist_ok=True)

        tensor = tensor.detach().cpu()

        B, C, H, W = tensor.shape

        for b in range(B):
            # Option 1: save channel-mean projection
            mean_proj = tensor[b].mean(dim=0)
            plt.imsave(os.path.join(save_path, f"{prefix}_b{b}_mean.png"),
                    mean_proj.numpy(), cmap='viridis')

            # Option 2: save first few channels
            for c in range(min(max_channels, C)):
                channel_img = tensor[b, c]
                plt.imsave(os.path.join(save_path, f"{prefix}_b{b}_c{c}.png"),
                        channel_img.numpy(), cmap='gray')



    def subtraction_loss(self, real_a, real_a_adj, gen_fwd, gen_bkw, step=None):
        """
        Compute subtraction consistency loss between adjacent slices using only the first channel.
        Also saves debug subtraction images for inspection.
        """

        # Use only the first channel (C=0) → shape: (B, H, W)
        z1 = real_a[:, 0, :, :]
        z2 = real_a_adj[:, 0, :, :]
        #print("DEBUG: z-spacing = :", self.z_spacing)
        subtract_real = torch.abs(z1 - z2) / self.z_spacing  # shape: (B, H, W)

        # Generate fake images
        fake_b = gen_fwd(real_a)
        fake_b_adj = gen_fwd(real_a_adj)
        recon_a = gen_bkw(fake_b)
        recon_a_adj = gen_bkw(fake_b_adj)

        # Use only first channel of generated images
        fake_z1 = recon_a[:, 0, :, :]
        fake_z2 = recon_a_adj[:, 0, :, :]

        subtract_fake = torch.abs(fake_z1 - fake_z2) /self.z_spacing

        #print(fake_b.shape, z1.shape, z2.shape, fake_z1.shape, fake_z2.shape, subtract_real.shape, subtract_fake.shape)
        if step % 100 == 0: 
            # Save subtraction images for debugging (just first sample)
            # Debug images. In practice, you might want to save these less frequently or only a few samples.
            os.makedirs(self.debug_root, exist_ok=True)
            # Only save first sample in batch

            z1_filename = self.images.real_a_names[0]  # string: "img=123_P=0.tif"
            z2_filename = self.images.real_a_adj_names[0]
            print(f"[Debug] z1 name: {z1_filename}, z2 name: {z2_filename}")
            save_image(fake_b[0],          os.path.join(self.debug_root, "fake_b_z.png"))
            save_image(z1[0],          os.path.join(self.debug_root, "z1_real.png"))
            save_image(z2[0],          os.path.join(self.debug_root, "z2_real.png"))
            save_image(fake_z1[0],     os.path.join(self.debug_root, "z1_fake.png"))
            save_image(fake_z2[0],     os.path.join(self.debug_root, "z2_fake.png"))
            save_image(subtract_real[0], os.path.join(self.debug_root, "subtraction_real.png"))
            save_image(subtract_fake[0], os.path.join(self.debug_root, "subtraction_fake.png"))

        # Compute L1 loss between subtraction maps
        return self.criterion_subtract(subtract_real, subtract_fake)


    def idt_forward_image(self, real, gen):
        # pylint: disable=no-self-use

        # (N, C, H, W)
        idt = gen(real)
        return idt
    
    # Function to save the forward images for debugging.
    def save_forward_image(self, real_a, real_a_adj, gen_ab, step=None):
        # pylint: disable=no-self-use

        # (N, C, H, W)
        fake_b = gen_ab(real_a)
        fake_b_adj = gen_ab(real_a_adj)
        if step % 100 == 0: 
            # Save subtraction images for debugging (just first sample)
            # Debug images. In practice, you might want to save these less frequently or only a few samples.
            os.makedirs(self.debug_root, exist_ok=True)
            save_image(real_a[0],          os.path.join(self.debug_root, "real_a.png"))
            save_image(real_a_adj[0],      os.path.join(self.debug_root, "real_a_adj.png"))
            save_image(fake_b[0],          os.path.join(self.debug_root, "fake_b.png"))
            save_image(fake_b_adj[0],      os.path.join(self.debug_root, "fake_b_adj.png"))

        return None

    def forward_dispatch(self, direction):
        if direction == 'ab':
            (
                self.images.fake_b, self.images.reco_a,
                self.images.consist_fake_b
            ) = self.cycle_forward_image(
                self.images.real_a, self.models.gen_ab, self.models.gen_ba
            )

        elif direction == 'ba':
            (
                self.images.fake_a, self.images.reco_b,
                self.images.consist_fake_a
            ) = self.cycle_forward_image(
                self.images.real_b, self.models.gen_ba, self.models.gen_ab
            )

        elif direction == 'aa':
            self.images.idt_a = \
                self.idt_forward_image(self.images.real_a, self.models.gen_ba)

        elif direction == 'bb':
            self.images.idt_b = \
                self.idt_forward_image(self.images.real_b, self.models.gen_ab)

        elif direction == 'avg-ab':
            (
                self.images.fake_b, self.images.reco_a,
                self.images.consist_fake_b
            ) = self.cycle_forward_image(
                self.images.real_a,
                self.models.avg_gen_ab, self.models.avg_gen_ba
            )

        elif direction == 'avg-ba':
            (
                self.images.fake_a, self.images.reco_b,
                self.images.consist_fake_a
            ) = self.cycle_forward_image(
                self.images.real_b,
                self.models.avg_gen_ba, self.models.avg_gen_ab
            )

        else:
            raise ValueError(f"Unknown forward direction: '{direction}'")

    def forward(self):
        if self.images.real_a is not None:
            if self.avg_momentum is not None:
                self.forward_dispatch(direction = 'avg-ab')
            else:
                self.forward_dispatch(direction = 'ab')

        if self.images.real_b is not None:
            if self.avg_momentum is not None:
                self.forward_dispatch(direction = 'avg-ba')
            else:
                self.forward_dispatch(direction = 'ba')


    def eval_consist_loss(
        self, consist_real_0, consist_fake_1, lambda_cycle_0
    ):
        return lambda_cycle_0 * self.lambda_consist * self.criterion_consist(
            consist_fake_1, consist_real_0
        )

    def eval_loss_of_cycle_forward(
        self, disc_1, real_0, fake_1, reco_0, fake_queue_1, lambda_cycle_0
    ):
        # pylint: disable=too-many-arguments
        # NOTE: Queue is updated in discriminator backprop
        disc_pred_fake_1 = queued_forward(
            disc_1, fake_1, fake_queue_1, update_queue = False
        )

        loss_gen   = self.criterion_gan(disc_pred_fake_1, True)
        loss_cycle = lambda_cycle_0 * self.criterion_cycle(reco_0, real_0)

        loss = loss_gen + loss_cycle

        return (loss_gen, loss_cycle, loss)

    def eval_loss_of_idt_forward(self, real_0, idt_0, lambda_cycle_0):
        loss_idt = (
              lambda_cycle_0
            * self.lambda_idt
            * self.criterion_idt(idt_0, real_0)
        )

        loss = loss_idt

        return (loss_idt, loss)

    def backward_gen(self, direction):
        if direction == 'ab':
            (self.losses.gen_ab, self.losses.cycle_a, loss) \
                = self.eval_loss_of_cycle_forward(
                    self.models.disc_b,
                    self.images.real_a, self.images.fake_b, self.images.reco_a,
                    self.queues.fake_b, self.lambda_a
                )

            if self.consist_model is not None:
                self.losses.consist_a = self.eval_consist_loss(
                    self.images.consist_real_a, self.images.consist_fake_b,
                    self.lambda_a
                )

                loss += self.losses.consist_a

             # ✅ Subtraction loss (adjacent z slices)
            if self.lambda_sub_loss > 0 and hasattr(self.images, "real_a_adj"):
                self.losses.subtraction_adj = self.subtraction_loss(
                    self.images.real_a,
                    self.images.real_a_adj,
                    self.models.gen_ab,
                    self.models.gen_ba,
                    step=self.current_step
                )

                loss += self.lambda_sub_loss * self.losses.subtraction_adj
            
            # Embedding Loss (adjacent z slices)
            if self.lambda_embedding_loss > 0 and hasattr(self.images, "real_a_adj"):
                self.losses.embedding_adj = self.embedding_loss(
                    self.images.real_a,
                    self.images.real_a_adj,
                    self.models.gen_ab,
                    self.models.gen_ba,
                    step=self.current_step
                )

                loss += self.lambda_embedding_loss * self.losses.embedding_adj
            
            # if running original UVCGAN2 without adjacent slice losses.  
            if self.lambda_embedding_loss == 0 and self.lambda_sub_loss == 0 and hasattr(self.images, "real_a_adj"):
                self.save_forward_image(self.images.real_a, self.images.real_a_adj, self.models.gen_ab, step=self.current_step)


        elif direction == 'ba':
            (self.losses.gen_ba, self.losses.cycle_b, loss) \
                = self.eval_loss_of_cycle_forward(
                    self.models.disc_a,
                    self.images.real_b, self.images.fake_a, self.images.reco_b,
                    self.queues.fake_a, self.lambda_b
                )

            if self.consist_model is not None:
                self.losses.consist_b = self.eval_consist_loss(
                    self.images.consist_real_b, self.images.consist_fake_a,
                    self.lambda_b
                )

                loss += self.losses.consist_b

        elif direction == 'aa':
            (self.losses.idt_a, loss) \
                = self.eval_loss_of_idt_forward(
                    self.images.real_a, self.images.idt_a, self.lambda_a
                )

        elif direction == 'bb':
            (self.losses.idt_b, loss) \
                = self.eval_loss_of_idt_forward(
                    self.images.real_b, self.images.idt_b, self.lambda_b
                )
        else:
            raise ValueError(f"Unknown forward direction: '{direction}'")


        loss.backward()

    def backward_discriminator_base(
        self, model, real, fake, queue_real, queue_fake
    ):
        # pylint: disable=too-many-arguments
        loss_gp = None

        if self.gp is not None:
            loss_gp = self.gp(
                model, fake, real,
                model_kwargs_fake = { 'extra_bodies' : queue_fake.query() },
                model_kwargs_real = { 'extra_bodies' : queue_real.query() },
            )
            loss_gp.backward()

        pred_real = queued_forward(
            model, real, queue_real, update_queue = True
        )
        loss_real = self.criterion_gan(pred_real, True)

        pred_fake = queued_forward(
            model, fake, queue_fake, update_queue = True
        )
        loss_fake = self.criterion_gan(pred_fake, False)

        loss = (loss_real + loss_fake) * 0.5
        loss.backward()

        return (loss_gp, loss)

    def backward_discriminators(self):
        fake_a = self.images.fake_a.detach()
        fake_b = self.images.fake_b.detach()

        loss_gp_b, self.losses.disc_b \
            = self.backward_discriminator_base(
                self.models.disc_b, self.images.real_b, fake_b,
                self.queues.real_b, self.queues.fake_b
            )

        if loss_gp_b is not None:
            self.losses.gp_b = loss_gp_b

        loss_gp_a, self.losses.disc_a = \
            self.backward_discriminator_base(
                self.models.disc_a, self.images.real_a, fake_a,
                self.queues.real_a, self.queues.fake_a
            )

        if loss_gp_a is not None:
            self.losses.gp_a = loss_gp_a

    def optimization_step_gen(self):
        self.set_requires_grad([self.models.disc_a, self.models.disc_b], False)
        self.optimizers.gen.zero_grad(set_to_none = True)

        dir_list = [ 'ab', 'ba' ]
        if self.lambda_idt > 0:
            dir_list += [ 'aa', 'bb' ]

        for direction in dir_list:
            self.forward_dispatch(direction)
            self.backward_gen(direction)

        self.optimizers.gen.step()

    def optimization_step_disc(self):
        self.set_requires_grad([self.models.disc_a, self.models.disc_b], True)
        self.optimizers.disc.zero_grad(set_to_none = True)

        self.backward_discriminators()

        self.optimizers.disc.step()

    def _accumulate_averages(self):
        update_average_model(
            self.models.avg_gen_ab, self.models.gen_ab, self.avg_momentum
        )
        update_average_model(
            self.models.avg_gen_ba, self.models.gen_ba, self.avg_momentum
        )

    def optimization_step(self):
        self.optimization_step_gen()
        self.optimization_step_disc()

        if self.avg_momentum is not None:
            self._accumulate_averages()
        
        self.current_step += 1

