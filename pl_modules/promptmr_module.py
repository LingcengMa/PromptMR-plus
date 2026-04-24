import torch
from data import transforms
from pl_modules import MriModule
from typing import List
import copy 
from mri_utils import SSIMLoss
import torch.nn.functional as F
import importlib

def get_model_class(module_name, class_name="PromptMR"):
    """
    Dynamically imports the specified module and retrieves the class.

    Args:
        module_name (str): The module to import (e.g., 'model.m1', 'model.m2').
        class_name (str): The class to retrieve from the module (default: 'PromptMR').

    Returns:
        type: The imported class.
    """
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    return model_class

class PromptMrModule(MriModule):

    def __init__(
        self,
        num_cascades: int = 12,
        num_adj_slices: int = 5,
        n_feat0: int = 48,
        feature_dim: List[int] = [72,96,120],
        prompt_dim: List[int] = [24,48,72],
        sens_n_feat0: int = 24,
        sens_feature_dim: List[int] = [36,48,60],
        sens_prompt_dim: List[int] = [12,24,36],
        len_prompt: List[int] = [5,5,5],
        prompt_size: List[int] = [64,32,16],
        n_enc_cab: List[int] = [2,3,3],
        n_dec_cab: List[int] = [2,2,3],
        n_skip_cab: List[int] = [1,1,1],
        n_bottleneck_cab: int = 3,
        no_use_ca: bool = False,
        learnable_prompt: bool = False,
        adaptive_input: bool = True,
        n_buffer: int = 4,
        n_history: int = 0,
        use_sens_adj: bool = True,
        model_version: str = "promptmr_v2",
        lr: float = 0.0002,
        lr_step_size: int = 11,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.01,
        complex_l2_weight: float = 0.0,
        use_checkpoint: bool = False,
        compute_sens_per_coil: bool = False,
        **kwargs,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational network.
            num_adj_slices: Number of adjacent slices.
            n_feat0: Number of top-level feature channels for PromptUnet.
            feature_dim: feature dim for each level in PromptUnet.
            prompt_dim: prompt dim for each level in PromptUnet.
            sens_n_feat0: Number of top-level feature channels for sense map
                estimation PromptUnet in PromptMR.
            sens_feature_dim: feature dim for each level in PromptUnet for
                sensitivity map estimation (SME) network.
            sens_prompt_dim: prompt dim for each level in PromptUnet in
                sensitivity map estimation (SME) network.
            len_prompt: number of prompt component in each level.
            prompt_size: prompt spatial size.
            n_enc_cab: number of CABs (channel attention Blocks) in DownBlock.
            n_dec_cab: number of CABs (channel attention Blocks) in UpBlock.
            n_skip_cab: number of CABs (channel attention Blocks) in SkipBlock.
            n_bottleneck_cab: number of CABs (channel attention Blocks) in BottleneckBlock.
            no_use_ca: not using channel attention.
            learnable_prompt: whether to set the prompt as learnable parameters.
            adaptive_input: whether to use adaptive input.
            n_buffer: number of buffer in adaptive input.
            n_history: number of historical feature aggregation, should be less than num_cascades.
            use_sens_adj: whether to use adjacent sensitivity map estimation.
            model_version: model version. Default is "promptmr_v2".
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
            complex_l2_weight: Weight for optional complex-domain L2 loss term.
            use_checkpoint: Whether to use checkpointing to trade compute for GPU memory.
            compute_sens_per_coil: (bool) whether to compute sensitivity maps per coil for memory saving
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.num_cascades = num_cascades
        self.num_adj_slices = num_adj_slices

        self.n_feat0 = n_feat0
        self.feature_dim = feature_dim
        self.prompt_dim = prompt_dim

        self.sens_n_feat0 = sens_n_feat0
        self.sens_feature_dim = sens_feature_dim
        self.sens_prompt_dim = sens_prompt_dim

        self.len_prompt = len_prompt
        self.prompt_size = prompt_size
        self.n_enc_cab = n_enc_cab
        self.n_dec_cab = n_dec_cab
        self.n_skip_cab = n_skip_cab
        self.n_bottleneck_cab = n_bottleneck_cab

        self.no_use_ca = no_use_ca

        self.learnable_prompt = learnable_prompt
        self.adaptive_input = adaptive_input
        self.n_buffer = n_buffer
        self.n_history = n_history
        self.use_sens_adj = use_sens_adj
        # two flags for reducing memory usage
        self.use_checkpoint = use_checkpoint
        self.compute_sens_per_coil = compute_sens_per_coil
        
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.complex_l2_weight = complex_l2_weight

        self.model_version = model_version
        PromptMR = get_model_class(f"models.{model_version}")  # Dynamically get the model class
        
        self.promptmr = PromptMR(
            num_cascades=self.num_cascades,
            num_adj_slices=self.num_adj_slices,
            n_feat0=self.n_feat0,
            feature_dim = self.feature_dim,
            prompt_dim = self.prompt_dim,
            sens_n_feat0=self.sens_n_feat0,
            sens_feature_dim = self.sens_feature_dim,
            sens_prompt_dim = self.sens_prompt_dim,
            len_prompt = self.len_prompt,
            prompt_size = self.prompt_size,
            n_enc_cab = self.n_enc_cab,
            n_dec_cab = self.n_dec_cab,
            n_skip_cab = self.n_skip_cab,
            n_bottleneck_cab = self.n_bottleneck_cab,
            no_use_ca=self.no_use_ca,
            learnable_prompt = learnable_prompt,
            n_history = self.n_history,
            n_buffer = self.n_buffer,
            adaptive_input = self.adaptive_input,
            use_sens_adj = self.use_sens_adj,
        )

        self.loss = SSIMLoss()

    @staticmethod
    def _center_crop_spatial_to_smallest(
        x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Center-crop two tensors to the same spatial size for real/complex layouts."""
        x_h, x_w = (x.shape[-3], x.shape[-2]) if (x.ndim >= 3 and x.shape[-1] == 2) else (x.shape[-2], x.shape[-1])
        y_h, y_w = (y.shape[-3], y.shape[-2]) if (y.ndim >= 3 and y.shape[-1] == 2) else (y.shape[-2], y.shape[-1])
        crop_h, crop_w = min(x_h, y_h), min(x_w, y_w)

        if x.ndim >= 3 and x.shape[-1] == 2:
            x = transforms.complex_center_crop(x, (crop_h, crop_w))
        else:
            x = transforms.center_crop(x, (crop_h, crop_w))
        if y.ndim >= 3 and y.shape[-1] == 2:
            y = transforms.complex_center_crop(y, (crop_h, crop_w))
        else:
            y = transforms.center_crop(y, (crop_h, crop_w))
        return x, y

    def configure_optimizers(self):

        optim = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # step lr scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )
        return [optim], [scheduler]
    
    def forward(self, masked_kspace, mask, num_low_frequencies, mask_type="cartesian", use_checkpoint=False, compute_sens_per_coil=False):
        return self.promptmr(masked_kspace, mask, num_low_frequencies, mask_type, use_checkpoint=use_checkpoint, compute_sens_per_coil=compute_sens_per_coil)   

    def training_step(self, batch, batch_idx):
        output_dict = self(batch.masked_kspace, batch.mask, batch.num_low_frequencies, batch.mask_type,
                           use_checkpoint=self.use_checkpoint, compute_sens_per_coil=self.compute_sens_per_coil)
        output = output_dict['img_pred']
        target, output = transforms.center_crop_to_smallest(
            batch.target, output)

        ssim_loss = self.loss(
            output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
        )
        loss = ssim_loss
        self.log("train_ssim_loss", ssim_loss, prog_bar=False)

        if self.complex_l2_weight > 0:
            pred_complex = output_dict.get("img_pred_complex", None)
            target_complex = batch.target
            if pred_complex is not None and target_complex.ndim >= 3 and target_complex.shape[-1] == 2:
                target_complex, pred_complex = self._center_crop_spatial_to_smallest(target_complex, pred_complex)
                complex_l2_loss = F.mse_loss(pred_complex, target_complex)
                loss = loss + self.complex_l2_weight * complex_l2_loss
                self.log("train_complex_l2_loss", complex_l2_loss, prog_bar=False)
            else:
                self.log("train_complex_l2_loss", torch.tensor(0.0, device=loss.device), prog_bar=False)

        self.log("train_loss", loss, prog_bar=True)

        ##! raise error if loss is nan
        if torch.isnan(loss):
            raise ValueError(f'nan loss on {batch.fname} of slice {batch.slice_num}')
        return loss

    def on_after_backward(self):
        if self.global_step % self.trainer.log_every_n_steps == 0:
            grads = [p.grad.detach() for p in self.promptmr.parameters() if p.grad is not None]
            if grads:
                grad_norm = torch.linalg.vector_norm(
                    torch.stack([torch.linalg.vector_norm(g, ord=2) for g in grads]),
                    ord=2,
                )
            else:
                grad_norm = torch.tensor(0.0, device=self.device)
            self.log("grad_norm", grad_norm)

    def validation_step(self, batch, batch_idx):

        output_dict = self(batch.masked_kspace, batch.mask, batch.num_low_frequencies, batch.mask_type,
                           compute_sens_per_coil=self.compute_sens_per_coil)
        output = output_dict['img_pred']
        img_zf = output_dict['img_zf']
        target, output = transforms.center_crop_to_smallest(
            batch.target, output)
        _, img_zf = transforms.center_crop_to_smallest(
            batch.target, img_zf)
        val_ssim_loss = self.loss(
                output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
            )
        val_loss = val_ssim_loss
        if self.complex_l2_weight > 0:
            pred_complex = output_dict.get("img_pred_complex", None)
            target_complex = batch.target
            if pred_complex is not None and target_complex.ndim >= 3 and target_complex.shape[-1] == 2:
                target_complex, pred_complex = self._center_crop_spatial_to_smallest(target_complex, pred_complex)
                val_complex_l2_loss = F.mse_loss(pred_complex, target_complex)
                val_loss = val_loss + self.complex_l2_weight * val_complex_l2_loss
        cc = batch.masked_kspace.shape[1]
        centered_coil_ksp_visual = torch.log10(1e-10+torch.view_as_complex(batch.masked_kspace[:,cc//2]).abs())
        centered_sens_maps_visual = output_dict['sens_maps'][:,cc//self.num_adj_slices//2].abs()
        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "img_zf":   img_zf,
            "mask": centered_coil_ksp_visual, 
            "sens_maps": centered_sens_maps_visual,
            "output": output,
            "target": target,
            "loss_ssim": val_ssim_loss,
            "loss": val_loss,
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output_dict = self(batch.masked_kspace, batch.mask, batch.num_low_frequencies, batch.mask_type,
                           compute_sens_per_coil=self.compute_sens_per_coil)
        output = output_dict['img_pred']

        crop_size = batch.crop_size 
        crop_size = [crop_size[0][0], crop_size[1][0]] # if batch_size>1
        # detect FLAIR 203
        if output.shape[-1] < crop_size[1]:
            crop_size = (output.shape[-1], output.shape[-1])
        output = transforms.center_crop(output, crop_size)

        num_slc = batch.num_slc
        return {
            'output': output.cpu(), 
            'slice_num': batch.slice_num, 
            'fname': batch.fname,
            'num_slc':  num_slc
        }
        
