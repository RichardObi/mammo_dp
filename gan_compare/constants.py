import logging
from dataclasses import dataclass
from typing import Optional
from torchvision import models
import torch.nn as nn
import torch
import os

from opacus.validators import ModuleValidator
from opacus import PrivacyEngine


from gan_compare.training.networks.classification.classifier_64 import Net as Net64
from gan_compare.training.networks.classification.classifier_128 import Net as Net128
from gan_compare.training.networks.classification.swin_transformer import (
    SwinTransformer,
)

# TODO: Consider refactoring into a constants.py and a separate model_utils.py

DATASET_LIST = ["bcdr", "inbreast", "cbis-ddsm"]
RADIMAGENET_SWIN_T_PATH = os.path.join("extension/swin_t_radimagenet","ckpt_epoch_274.pth") # Default
RADIMAGENET_SWIN_T_PATH_2 = os.path.join("extension/swin_t_radimagenet","rin_swintf.pth")
RADIMAGENET_SWIN_T_PATH_3 = os.path.join("extension/swin_t_radimagenet","img2rin_swintf.pth")


swin_transformer_names = ["swin_transformer", "swin_t_scratch_frozen", "swin_t_imagenet", "swin_t_imagenet_frozen", "swin_t_radimagenet", "swin_t_radimagenet_frozen"]


def get_radimagenet_weights_path(config, pretrained_weights_path):
    if pretrained_weights_path is None: pretrained_weights_path = RADIMAGENET_SWIN_T_PATH
    if config.radimagenet_weights == 2:
        pretrained_weights_path = RADIMAGENET_SWIN_T_PATH_2
    elif config.radimagenet_weights == 3:
        pretrained_weights_path = RADIMAGENET_SWIN_T_PATH_3
    return pretrained_weights_path


def get_classifier(config: dataclass, num_classes: Optional[int] = None, device=None, weights_path=None) -> nn.Module:

    if not config.image_size == 224:
        logging.warning(
            f"For {config.model_name}, you are using image_size {config.image_size}, while the default image shape is 224x224x3."
        )

    if num_classes is None:
        # FIXME During GAN training config.n_cond is the number of conditions and not the number of classes.
        # Workaround: Usage of value from num_classes attribute instead.
        num_classes = config.num_classes
    reinit_head = True if weights_path is None else False
    if config.model_name == "swin_transformer":
        if config.pretrained_mode is not None:
            # Load custom pre-trained weights:
            swin_t_model = SwinTransformer(num_classes=1000, img_size=config.image_size)
        else:
            swin_t_model = SwinTransformer(num_classes=num_classes, img_size=config.image_size)
        swin_t_model = load_pretrained_weights(config=config, weights_path=weights_path, net=swin_t_model, device=device)
        return swin_t_model

    elif config.model_name == "swin_t_scratch_frozen":
        swin_t_model = SwinTransformer(num_classes=1000, img_size=config.image_size)
        swin_t_model = load_pretrained_weights(config=config, weights_path= weights_path, net=swin_t_model, freeze=True, pretrained_mode="swin_t_frozen", device=device)
        swin_t_model = freeze_layers(config=config, net=swin_t_model, pretrained_mode="swin_t_frozen", reinit_head=reinit_head) # in theory this line is not necessary
        return swin_t_model

    elif config.model_name == "swin_t_imagenet":
        swin_t_model = models.swin_t(weights='IMAGENET1K_V1') # img_size per default = 224
        logging.debug(swin_t_model)
        is_frozen = True if config.pretrained_mode=='freeze_all' else False # in this case we freeze all layers (apart from the head)
        swin_t_model = load_pretrained_weights(config=config, weights_path=weights_path, net=swin_t_model, device=device, freeze=is_frozen)
        # Now reinit last layer with only 2 instead of 1000 classes
        if config.reinit_head is not None:
            reinit_head = config.reinit_head
        if reinit_head: swin_t_model.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)
        return swin_t_model

    elif config.model_name == "swin_t_radimagenet":
        swin_t_model = SwinTransformer(num_classes=1000,
                                       img_size=config.image_size) if config.radimagenet_weights == 1 else SwinTransformer(num_classes=165, img_size=config.image_size)
        is_frozen = True if config.pretrained_mode=='freeze_all' else False # in this case we freeze all layers (apart from the head)

        config.pretrained_weights_path = get_radimagenet_weights_path(config, weights_path)
        swin_t_model = load_pretrained_weights(config=config, net=swin_t_model, weights_path=weights_path,
                                               freeze=is_frozen, device=device)
        # Now reinit last layer with only 2 instead of 1000 classes
        if config.reinit_head is not None:
            reinit_head = config.reinit_head
        if reinit_head: swin_t_model.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)
        return swin_t_model

    elif config.model_name == "swin_t_imagenet_frozen" or config.model_name == "swin_t_radimagenet_frozen":
        if config.model_name == "swin_t_imagenet_frozen":
            swin_t_model = models.swin_t(weights='IMAGENET1K_V1') # img_size per default = 224
        elif config.model_name == "swin_t_radimagenet_frozen":
            swin_t_model = SwinTransformer(num_classes=1000, img_size=config.image_size) if config.radimagenet_weights == 1 else SwinTransformer(num_classes=165, img_size=config.image_size)
            config.pretrained_weights_path = get_radimagenet_weights_path(config, weights_path)
        swin_t_model = load_pretrained_weights(config=config, net=swin_t_model, weights_path=weights_path, freeze=False, pretrained_mode="swin_t_frozen", device=device)
        # freeze under consideration of reinit_head
        if config.reinit_head is not None:
            # this is repeated inside freeze_layers() <- refactor this
            reinit_head = config.reinit_head
        swin_t_model = freeze_layers(config=config, net=swin_t_model, pretrained_mode="swin_t_frozen", reinit_head=reinit_head)
        return swin_t_model

    elif config.model_name == "cnn":
        return_probabilities = (
            False
            if hasattr(config, "pretrain_classifier") is False
            else config.pretrain_classifier
        )
        if config.image_size == 64:
            return Net64(
                num_labels=num_classes, return_probabilities=return_probabilities
            )
        elif config.image_size == 128:
            return Net128(
                num_labels=num_classes, return_probabilities=return_probabilities
            )
        raise ValueError(f"Unrecognized CNN image size = {config.image_size}")
    raise ValueError(f"Unrecognized model name = {config.name}")


# These are the layers with trainable weights in W-MSA, SW-MSA, and MLP:
freeze_template_swin_transformer_almost_all_layers = [
    "layers.0.blocks.0.attn.qkv",
    "layers.0.blocks.0.attn.proj",
    "layers.0.blocks.0.mlp.fc1",
    "layers.0.blocks.0.mlp.fc2",
    "layers.0.blocks.1.attn.qkv",
    "layers.0.blocks.1.attn.proj",
    "layers.0.blocks.1.mlp.fc1",
    "layers.0.blocks.1.mlp.fc2",
    "layers.1.blocks.0.attn.qkv",
    "layers.1.blocks.0.attn.proj",
    "layers.1.blocks.0.mlp.fc1",
    "layers.1.blocks.0.mlp.fc2",
    "layers.1.blocks.1.attn.qkv",
    "layers.1.blocks.1.attn.proj",
    "layers.1.blocks.1.mlp.fc1",
    "layers.1.blocks.1.mlp.fc2",
    "layers.2.blocks.0.attn.qkv",
    "layers.2.blocks.0.attn.proj",
    "layers.2.blocks.0.mlp.fc1",
    "layers.2.blocks.0.mlp.fc2",
    "layers.2.blocks.1.attn.qkv",
    "layers.2.blocks.1.attn.proj",
    "layers.2.blocks.1.mlp.fc1",
    "layers.2.blocks.1.mlp.fc2",
    "layers.2.blocks.2.attn.qkv",
    "layers.2.blocks.2.attn.proj",
    "layers.2.blocks.2.mlp.fc1",
    "layers.2.blocks.2.mlp.fc2",
    "layers.2.blocks.3.attn.qkv",
    "layers.2.blocks.3.attn.proj",
    "layers.2.blocks.3.mlp.fc1",
    "layers.2.blocks.3.mlp.fc2",
    "layers.2.blocks.4.attn.qkv",
    "layers.2.blocks.4.attn.proj",
    "layers.2.blocks.4.mlp.fc1",
    "layers.2.blocks.4.mlp.fc2",
    "layers.2.blocks.5.attn.qkv",
    "layers.2.blocks.5.attn.proj",
    "layers.2.blocks.5.mlp.fc1",
    "layers.2.blocks.5.mlp.fc2",
    "layers.3.blocks.0.attn.qkv",
    "layers.3.blocks.0.attn.proj",
    "layers.3.blocks.0.mlp.fc1",
    "layers.3.blocks.0.mlp.fc2",
    "layers.3.blocks.1.attn.qkv",
    "layers.3.blocks.1.attn.proj",
    "layers.3.blocks.1.mlp.fc1",
    "layers.3.blocks.1.mlp.fc2",
]

def load_pretrained_weights(config, net, weights_path=None, freeze=True, pretrained_mode=None, device=None):
    if device == None:
        device = torch.device(
            "cuda" if (torch.cuda.is_available() and config.ngpu > 0) else "cpu"
        )

    reinit_head = True if weights_path is None else False
    if config.pretrained_weights_path is not None and weights_path is None:
        logging.info(f"Replacing path to pretrained weights 'weights_path' ({weights_path}) with 'config.pretrained_weights_path' ({config.pretrained_weights_path}).")
        weights_path = config.pretrained_weights_path
    if "swin_transformer" in config.model_name.lower():
        # this produces errors when loading Swin T Imagenet from pytorch (different layer names)
        # We have a swin transformer model that we pretrained on binary CLF. Let's re-init model with correct num of classes.
        net = SwinTransformer(num_classes=2, img_size=config.image_size)
    elif config.reinit_head is False:
        # FIXME THis will only work if the pretrained model has only one linear head layer.
        net.head = nn.Linear(in_features=768, out_features=config.num_classes, bias=True)
        logging.warning(f"We are initializing a new single layer nn.Linear head to be able to load the checkpoint. In case config.reinit_head=={config.reinit_head} is True, we later try to copy pretrained head weights into the model {config.model_name.lower()} from weights_path {weights_path}.")
    elif weights_path is not None and ("frozen" in config.model_name.lower()):
        # This is only if we have already trained a frozen model and want to only test with that model.
        # Else we might have the case where we want to load weights and afterwards change the last layer in frozen setting (e.g. for further training)
        logging.warning(f"We are initializing a new head to be able to load the checkpoint. In case config.reinit_head=={config.reinit_head} is True, we later try to copy pretrained head weights into the model {config.model_name.lower()} from weights_path {weights_path}.")
        net.head = nn.Sequential(
            nn.Linear(in_features=768, out_features=384, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=384, out_features=config.num_classes, bias=True)
        )
    elif reinit_head:
        # pass ensures that the net.head is the same (e.g. same number of outputs) as in the pretrained model that we load further below in this function.
        # This avoid the problem of loading the state_dict with an existing head. The final net.head will be reinitialized at a later stage.
        pass
    else:
        # We are here if we want to load the pretrained weights we trained ourselves (e.g. during synthetic data pretraining).
        # Only reinit the last layer, leaving everything else in "net" as is.
        logging.warning(f"Reinit head: {reinit_head} to nn.Linear(in_features=768, out_features=config.num_classes, bias=True)")
        net.head = nn.Linear(in_features=768, out_features=config.num_classes, bias=True)

    logging.info(f"Now loading pretrained weights on {device} from path: {weights_path}. Freezing layers(?): {freeze}. pretrained_mode: {pretrained_mode}")

    if weights_path is None or not os.path.exists(weights_path) or not os.path.isfile(weights_path):
        logging.warning(f"Tried loading pretrained weights from {weights_path}. This path seems to not be valid. Please revise if this is not desired behaviour. Fallback: Now continuing WITHOUT loading pretrained model.")
    else:
        pretrained_dict = torch.load(weights_path, map_location=device)
        if 'model' in list(pretrained_dict.keys()):
            pretrained_dict = pretrained_dict["model"]
        try:
            net.load_state_dict(pretrained_dict)
        except Exception as e:
            # Error if loaded state_dict has keys that are not part of net's key. Let's remove these values and try again.
            # https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3
            logging.warning(f"WARNING!!!!!! Now adjusting pretrained state_dict automatically. ")
            logging.warning(f"Adjusting pretrained state_dict and trying again, as error occurred during standard state_dict loading: {e} ")
            net_dict = net.state_dict()
            logging.debug(f"net.head: {net.head}")
            logging.debug(f"pretrained_dict: {pretrained_dict}")

            # Filter out unnecessary keys in pretrained net
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
            # Overwrite entries in the existing state dict
            net_dict.update(pretrained_dict)
            # Initialise model parameters based on new adjusted state dict
            net.load_state_dict(net_dict)

        # In case some layers are to be frozen (i.e. not further trained via stochastic gradient descent).
    if freeze:
        net = freeze_layers(config, net, pretrained_mode, reinit_head=reinit_head)
    return net

def freeze_layers(config, net, pretrained_mode = None, num_classes=None, reinit_head=False):
    if config.reinit_head is not None:
        reinit_head = config.reinit_head
    if num_classes is None:
        num_classes = config.num_classes
    if pretrained_mode is not None and pretrained_mode == "swin_t_frozen" or config.pretrained_mode == "freeze_all":
        for param in net.parameters():
            # Freezing all layers of Swin-T Transformer
            param.requires_grad = False
        # Adding two FCN linear layers as prediction heads. Use Swin-T as feature extractor
        # Now re-init the layers (e.g. last few layers) that will be not frozen / trainable
        # Re-init with only 2 instead of 1000 classes
        # swin_t_model.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)
        if reinit_head:
            last_layers = nn.Sequential(
                nn.Linear(in_features=768, out_features=384, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(in_features=384, out_features=num_classes, bias=True)
            )
            net.head = last_layers
        else:
            # If we do not reinit the head, we assume that the head is already correctly set up and need to set it
            # trainable again after freezing other params
            net.head.requires_grad_(True)
            logging.info(f"head was not reinitialized, but set to trainable again.")

    elif config.pretrained_mode == "freeze_specific_weights":
        #layers_to_be_frozen = getattr(constants, config.pretrained_freeze_template)
        layers_to_be_frozen = freeze_template_swin_transformer_almost_all_layers
        logging.info(
            f"Freezing all weights in the following layers: {layers_to_be_frozen}"
        )
        for layer_string in layers_to_be_frozen:
            level_o = net
            levels = layer_string.split(".")
            for level_s in levels:
                level_o = (
                    level_o[int(level_s)]
                    if level_s.isnumeric()
                    else getattr(level_o, level_s)
                )
            for param in level_o.parameters():
                param.requires_grad = False
    elif config.pretrained_mode == "freeze_no_weights":
        pass  # nothing to do, because no freezing required and all weights have been loaded
    elif config.pretrained_mode is None:
        logging.warning(
            f"No information was provided as to which layers to freeze (config.pretrained_mode={config.pretrained_mode}). Fallback: Continuing without freezing any layers."
        )
    else:
        raise Exception(
            f"Value of config variable 'pretrained_mode' unknown: {config.pretrained_mode}"
        )
    return net


def make_private_with_epsilon(net, optimizer, dataloader, dp_target_epsilon, dp_target_delta, dp_max_grad_norm, num_epochs, auto_validate_model = False, grad_sample_mode=None, accountant = "rdp"):
    # Validate model
    errors = ModuleValidator.validate(net, strict=False)
    if errors is not None:
        logging.warning(f"DP errors: {errors}")
    if auto_validate_model:
        net = ModuleValidator.fix(net)
    privacy_engine = PrivacyEngine(accountant = accountant, secure_mode = False) #accountant: str = "prv", "gdp", "rdp"
    if grad_sample_mode is None: # Default
        net, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=net,
            optimizer=optimizer,
            data_loader=dataloader,
            epochs=num_epochs,
            target_epsilon=dp_target_epsilon, #hyperparam 1
            target_delta=dp_target_delta,  #hyperparam 2
            max_grad_norm=dp_max_grad_norm,  #hyperparam 3
            #grad_sample_mode="ew" # https://github.com/pytorch/opacus/issues/454
        )
    else:
        net, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=net,
            optimizer=optimizer,
            data_loader=dataloader,
            epochs=num_epochs,
            target_epsilon=dp_target_epsilon, #hyperparam 1
            target_delta=dp_target_delta,  #hyperparam 2
            max_grad_norm=dp_max_grad_norm,  #hyperparam 3
            grad_sample_mode="ew" # https://github.com/pytorch/opacus/issues/454
        )
    logging.info(f"Integrated Differential Privacy: PrivacyEngine for target DP(ε={dp_target_epsilon},δ={dp_target_delta}) with max_grad_norm={dp_max_grad_norm}.")
    # make sure return order is correct.
    return net, optimizer, train_loader, privacy_engine



def only_use_opacus_supported_layers(net):
    errors = ModuleValidator.validate(module=net, strict=False)
    logging.warning(f"Now fixing network in case of detected DP errors: {errors}")
    return ModuleValidator.fix(module=net)
