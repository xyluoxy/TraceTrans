from . import networks, losses
import torch
import torch.nn as nn
import os

class TraceTransModel:
    '''
    Top-level orchestrator for training and inference of TraceTrans.

    Holds the whole system:

    * ``netG`` -- a :class:`networks.DualStreamBackbone` generator that jointly predicts a
      translated image (``gen_y``) and a deformation field (``flow``) used to warp the
      source into ``warped_x``.
    * ``netD_gen`` / ``netD_warp`` -- two PatchGAN discriminators (only built when
      ``is_train``) that score the translated image and the warped image respectively,
      each conditioned on the source.
    * losses -- :class:`losses.SoftMutualInformation` for same-/cross-domain similarity,
      :class:`losses.Grad` for flow smoothness, :class:`losses.GANLoss` for the
      adversarial term, plus an L1 reconstruction term.
    * Adam optimizers for the generator and each discriminator.

    ``optimize_parameters`` runs one alternating GAN step (generator then discriminators)
    with a WGAN-GP-style gradient penalty and gradient clipping. ``gen_proportion``
    balances the generation stream against the warping stream in the combined losses.

    Parameters:
        opt: configuration object with fields such as ``gpu``, ``input_nc``,
            ``output_nc``, ``ndf``, ``net_D``, ``use_depth``, ``integration_steps``,
            ``gan_mode``, ``lr_G`` / ``lr_D``, ``beta_G`` / ``beta_D``.
        is_train (bool): when ``False``, only the generator is created (inference mode).
    '''
    def __init__(self, opt, is_train=True):
        self.opt = opt
        self.is_train = is_train
        self.device = f"cuda:{opt.gpu[0]}" if len(opt.gpu) > 0 else "cpu"
        if(self.device != 'cpu'): 
            assert(torch.cuda.is_available())
        
        self.netG = networks.DualStreamBackbone(
            ndim=2,
            in_channels=opt.input_nc if not opt.use_depth else opt.input_nc+1,
            out_channels=opt.output_nc,
            nb_features=(64, 128, 256, 512, 512),
            normalizations=nn.InstanceNorm2d,
            integration_steps=opt.integration_steps,
            device=self.device
        )

        if self.is_train:
            self.netD_gen = networks.define_D(
                input_nc=opt.input_nc*2 if not opt.use_depth else (opt.input_nc+1)*2,
                ndf=opt.ndf,
                netD=opt.net_D,
                device=self.device,
            )
            self.netD_warp = networks.define_D(
                input_nc=opt.input_nc*2 if not opt.use_depth else (opt.input_nc+1)*2,
                ndf=opt.ndf,
                netD=opt.net_D,
                device=self.device,
            )

            # self.SimLoss = losses.NCC().loss
            self.SimLoss = losses.SoftMutualInformation(num_bins=16).loss
            self.cross_domain_SimLoss = losses.SoftMutualInformation(num_bins=32).loss
            self.GradLoss = losses.Grad().loss
            self.GANLoss = losses.GANLoss(gan_mode = opt.gan_mode)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr = opt.lr_G, betas = (opt.beta_G, 0.999))
            self.optimizer_D_gen = torch.optim.Adam(self.netD_gen.parameters(), lr = opt.lr_D, betas = (opt.beta_D, 0.999))
            self.optimizer_D_warp = torch.optim.Adam(self.netD_warp.parameters(), lr = opt.lr_D, betas = (opt.beta_D, 0.999))


    def set_input(self, source, target, gtwarped):
        """
            Store one batch of training data on the generator's device.

            Parameters:
                source: source (moving) image batch -> ``self.real_x``
                target: target (fixed) image batch -> ``self.real_y``
                gtwarped: ground-truth warped source batch -> ``self.real_warped``
        """
        self.real_x = source.to(self.netG.device)
        self.real_y = target.to(self.netG.device)
        self.real_warped = gtwarped.to(self.netG.device)
        

    def forward(self):
        """
            Run the generator on ``self.real_x`` and cache its outputs.

            Sets ``self.gen_y`` (translated image), ``self.warped_x`` (warped source) and
            ``self.flow``. In training mode (``is_train``) ``self.flow`` is the
            pre-integration field used for the smoothness loss; otherwise it is the
            integrated deformation field.
        """
        result_gen, result_warp = self.netG(self.real_x, register=(not self.is_train))
        self.gen_y = result_gen
        self.warped_x, self.flow = result_warp

    def predict(self):
        """
            Inference-only forward: return ``(gen_y, [warped_x, integrated_flow])`` for
            ``self.real_x`` without tracking gradients.
        """
        with torch.no_grad():
            return self.netG(self.real_x, register=True)

    def backward_D(self, lambda_adv, gen_proportion):
        """
            Compute and backprop the discriminator loss for both streams.

            For ``netD_gen`` and ``netD_warp`` this sums the real/fake hinge terms plus a
            WGAN-GP gradient penalty, weights them by ``gen_proportion`` /
            ``1 - gen_proportion``, scales the total by ``lambda_adv``, and clips the
            discriminator gradients. Fake inputs are detached so no gradient reaches
            ``netG``.

            Parameters:
                lambda_adv (float): overall weight of the adversarial term.
                gen_proportion (float): weight of the generation stream vs the warping
                    stream, in ``[0, 1]``.

            Returns:
                dict: scalar loss components for logging.
        """
        real_condition_gen = torch.cat((self.real_x, self.real_y), dim=1)
        pred_real_gen = self.netD_gen(real_condition_gen)
        loss_D_gen_real = self.GANLoss(pred_real_gen, True)

        # since this is paired training, no fake_pool is needed
        
        fake_condition_gen = torch.cat((self.real_x, self.gen_y), dim=1)
        # detach gen_y from compute graph to avoid gradient flowing to netG
        pred_fake_gen = self.netD_gen(fake_condition_gen.detach())
        loss_D_gen_fake = self.GANLoss(pred_fake_gen, False)
        
        real_condition_warp = torch.cat((self.real_x, self.real_y), dim=1)
        pred_real_warp = self.netD_warp(real_condition_warp)
        loss_D_warp_real = self.GANLoss(pred_real_warp, True)
        
        fake_condition_warp = torch.cat((self.real_x, self.warped_x), dim=1)
        # detach warped_x compute graph to avoid gradient flowing to netG
        pred_fake_warp = self.netD_warp(fake_condition_warp.detach())
        loss_D_warp_fake = self.GANLoss(pred_fake_warp, False)

        # Add gradient penalty for discriminator regularization
        alpha = torch.rand(real_condition_gen.size(0), 1, 1, 1).to(self.device)
        
        # Gradient penalty for netD_gen
        interpolated_gen = alpha * real_condition_gen + (1 - alpha) * fake_condition_gen.detach()
        interpolated_gen.requires_grad_(True)
        pred_interpolated_gen = self.netD_gen(interpolated_gen)
        gradients_gen = torch.autograd.grad(
            outputs=pred_interpolated_gen,
            inputs=interpolated_gen,
            grad_outputs=torch.ones_like(pred_interpolated_gen),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient_penalty_gen = torch.mean((gradients_gen.norm(2, dim=1) - 1) ** 2)
        
        # Gradient penalty for netD_warp
        interpolated_warp = alpha * real_condition_warp + (1 - alpha) * fake_condition_warp.detach()
        interpolated_warp.requires_grad_(True)
        pred_interpolated_warp = self.netD_warp(interpolated_warp)
        gradients_warp = torch.autograd.grad(
            outputs=pred_interpolated_warp,
            inputs=interpolated_warp,
            grad_outputs=torch.ones_like(pred_interpolated_warp),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient_penalty_warp = torch.mean((gradients_warp.norm(2, dim=1) - 1) ** 2)

        warp_proportion = 1 - gen_proportion
        loss_D_gen = gen_proportion * ((loss_D_gen_real + loss_D_gen_fake) * 0.5)
        loss_D_warp = warp_proportion * ((loss_D_warp_real + loss_D_warp_fake) * 0.5)
        loss_D = loss_D_gen + loss_D_warp
        loss_D = loss_D + 2 * (gen_proportion * gradient_penalty_gen + warp_proportion * 3 * gradient_penalty_warp)
        
        loss_dict = {
            'loss_D': loss_D.item(),
            'loss_D_gen': ((loss_D_gen_real + loss_D_gen_fake) * 0.5).item(),
            'loss_D_warp': ((loss_D_warp_real + loss_D_warp_fake) * 0.5).item(),
            'gradient_penalty': ((gradient_penalty_gen + gradient_penalty_warp) * 0.5).item(),
            'gradient_penalty_gen': gradient_penalty_gen.item(),
            'gradient_penalty_warp': gradient_penalty_warp.item()
        }
        
        loss_D = lambda_adv * loss_D
        loss_D.backward()
        
        # Add gradient clipping for discriminator
        torch.nn.utils.clip_grad_norm_(self.netD_gen.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.netD_warp.parameters(), max_norm=1.0)
        
        return loss_dict
    
    def backward_G(self, lambda_adv, gen_proportion):
        """
            Compute and backprop the generator loss, then clip its gradients.

            Combines: an L1 reconstruction term on ``gen_y``, soft-MI similarity on the
            warped source (and on depth when ``opt.use_depth``), a cross-domain soft-MI
            term between the generated and warped images, the flow smoothness penalty
            (:class:`losses.Grad`), and the adversarial term (weighted by ``lambda_adv``).
            The generation and warping contributions are mixed by ``gen_proportion``.

            Parameters:
                lambda_adv (float): overall weight of the adversarial term.
                gen_proportion (float): generation vs warping stream weight, in ``[0, 1]``.

            Returns:
                dict: scalar loss components for logging (``loss_G`` is the total).
        """
        fake_condition_gen = torch.cat((self.real_x, self.gen_y), dim=1)
        pred_fake_gen = self.netD_gen(fake_condition_gen)
        loss_D_gen = self.GANLoss(pred_fake_gen, True)
        
        fake_condition_warp = torch.cat((self.real_x, self.warped_x), dim=1)
        pred_fake_warp = self.netD_warp(fake_condition_warp)
        loss_D_warp = self.GANLoss(pred_fake_warp, True)
        
        warp_proportion = 1 - gen_proportion
        loss_D = gen_proportion * loss_D_gen + warp_proportion * loss_D_warp
        
        real_y_texture = self.real_y if not self.opt.use_depth else self.real_y[:, :-1, :, :]
        warped_x_texture = self.warped_x if not self.opt.use_depth else self.warped_x[:, :-1, :, :]
        # real_warped_texture = self.real_warped[:, :1, :, :]
        
        # Reconstruction loss (L1 loss for pixel-to-pixel)
        loss_L1 = torch.mean(torch.abs(self.gen_y - self.real_y))
        # Similarity loss
        loss_Sim_gen = 10 * loss_L1
        loss_Sim_warp = self.SimLoss(real_y_texture, warped_x_texture)
        loss_Sim_cross_domain = self.cross_domain_SimLoss(self.gen_y[:, :1, :, :], warped_x_texture)
        loss_Sim = ((gen_proportion * loss_Sim_gen 
                     + warp_proportion * loss_Sim_warp 
                     + min(gen_proportion, warp_proportion) * 0.25 * loss_Sim_cross_domain)
                     / (gen_proportion + warp_proportion + min(gen_proportion, warp_proportion)))
            
        loss_Grad = self.GradLoss(None, self.flow)
        
        loss_dict = {
            'loss_G': None,
            'loss_Sim': loss_Sim.item(),
            'loss_Sim_gen': loss_L1.item(),
            'loss_Sim_warp': loss_Sim_warp.item(),
            'loss_Sim_cross_domain': loss_Sim_cross_domain.item(),
            'loss_Grad': loss_Grad.item(),
            'loss_Depth': None,
            'loss_D': loss_D.item(),
            'loss_D_gen': loss_D_gen.item(),
            'loss_D_warp': loss_D_warp.item()
        }
        
        if self.opt.use_depth:
            real_y_depth = self.real_y[:, 1:, :, :]
            warped_x_depth = self.warped_x[:, 1:, :, :]
            loss_Depth = self.SimLoss(real_y_depth, warped_x_depth)
            loss_dict['loss_Depth'] = loss_Depth.item()
        
        grad_norm = 0.1 * (warp_proportion / 0.5)
        
        if not self.opt.use_depth:
            loss_G = loss_Sim + lambda_adv * loss_D + grad_norm * loss_Grad
        else:
            loss_G = loss_Sim + 0.3 * loss_Depth + grad_norm * loss_Grad + lambda_adv * loss_D
            
        loss_dict['loss_G'] = loss_G.item()
        loss_G.backward()
        
        # Add gradient clipping for generator
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=1.0)
        
        return loss_dict
    
    def optimize_parameters(self, lambda_adv, gen_proportion=0.5):
        """
            Run one full training step: forward pass, generator update (with the
            discriminators frozen), then discriminator update.

            Parameters:
                lambda_adv (float): overall weight of the adversarial term.
                gen_proportion (float): generation vs warping stream weight, in ``[0, 1]``
                    (default ``0.5``); out-of-range values raise ``ValueError``.

            Returns:
                tuple[dict, dict]: ``(loss_dict_G, loss_dict_D)``.
        """
        if gen_proportion < 0 or gen_proportion > 1:
            raise ValueError(f"gen_proportion must stay between 0 and 1, but got '{gen_proportion}'")
        
        self.forward()
        
        self.set_requires_grad(self.netD_gen, False)
        self.set_requires_grad(self.netD_warp, False)
        self.optimizer_G.zero_grad()
        loss_dict_G = self.backward_G(lambda_adv, gen_proportion)
        self.optimizer_G.step()
        
        self.set_requires_grad(self.netD_gen, True)
        self.set_requires_grad(self.netD_warp, True)
        self.optimizer_D_gen.zero_grad()
        self.optimizer_D_warp.zero_grad()
        loss_dict_D = self.backward_D(lambda_adv, gen_proportion)
        self.optimizer_D_gen.step()
        self.optimizer_D_warp.step()
        
        return loss_dict_G, loss_dict_D

    def set_requires_grad(self, nets, requires_grad=False):
        """
            Toggle ``requires_grad`` on every parameter of the given network(s), used to
            freeze the discriminators while the generator is updated (and vice versa).

            Parameters:
                nets (nn.Module | list[nn.Module]): a network or list of networks
                    (``None`` entries are skipped).
                requires_grad (bool): target flag value. Default ``False``.
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def save(self, save_path):
        """Save ``netG`` (and, in training mode, both discriminators) as separate
        ``state_dict`` files under ``save_path``."""
        torch.save(self.netG.state_dict(), os.path.join(save_path, "netG"))
        torch.save(self.netD_gen.state_dict(), os.path.join(save_path, "netD_gen"))
        torch.save(self.netD_warp.state_dict(), os.path.join(save_path, "netD_warp"))

    def load(self, ckpt_path):
        """Load ``netG`` from ``ckpt_path``; in training mode also load ``netD_gen`` and
        ``netD_warp``. Expects the file layout produced by :meth:`save`."""
        assert os.path.exists(ckpt_path), f"checkpoint path '{ckpt_path}' does not exist"
        self.netG.load_state_dict(torch.load(os.path.join(ckpt_path, "netG")))
        if(self.is_train): 
            self.netD_gen.load_state_dict(torch.load(os.path.join(ckpt_path, "netD_gen")))
            self.netD_warp.load_state_dict(torch.load(os.path.join(ckpt_path, "netD_warp")))