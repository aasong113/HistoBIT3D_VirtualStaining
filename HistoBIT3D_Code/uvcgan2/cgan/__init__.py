from .cyclegan           import CycleGANModel
from .pix2pix            import Pix2PixModel
from .autoencoder        import Autoencoder
from .simple_autoencoder import SimpleAutoencoder
from .uvcgan2            import UVCGAN2
from .uvcgan2_3D_subtraction_loss         import UVCGAN2_3D_subtraction_loss  
from .uvcgan2_3D_embedding_loss         import UVCGAN2_3D_embedding_loss
from .uvcgan2_3D_emb_sub_stylefusion import UVCGAN2_3D_stylefusion
from .uvcgan2_3D_emb_sub_style_content import UVCGAN2_3D_emb_sub_style_content


CGAN_MODELS = {
    'cyclegan'           : CycleGANModel,
    'pix2pix'            : Pix2PixModel,
    'autoencoder'        : Autoencoder,
    'simple-autoencoder' : SimpleAutoencoder,
    'uvcgan2'            : UVCGAN2,
    'uvcgan2_3D_subtraction_loss'         : UVCGAN2_3D_subtraction_loss,
    'uvcgan2_3D_embedding_loss'         : UVCGAN2_3D_embedding_loss,
    'uvcgan2_3D_stylefusion'         : UVCGAN2_3D_stylefusion,
    'uvcgan2_3D_emb_sub_style_content'         : UVCGAN2_3D_emb_sub_style_content,

}

def select_model(name, **kwargs):
    if name not in CGAN_MODELS:
        raise ValueError("Unknown model: %s" % name)

    return CGAN_MODELS[name](**kwargs)

def construct_model(savedir, config, is_train, device):
    model = select_model(
        config.model, savedir = savedir, config = config, is_train = is_train,
        device = device, **config.model_args
    )

    return model
