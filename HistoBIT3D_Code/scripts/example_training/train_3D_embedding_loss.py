import argparse
import sys
import os
# Go up 3 levels to get repo root from inside scripts/2025*/train_3D.py
new_repo_root = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
# Remove old UVCGAN paths and add the correct one
sys.path = [p for p in sys.path if 'UVCGANv2_vHE' not in p or new_repo_root in os.path.abspath(p)]
sys.path.insert(0, new_repo_root)
print("Using uvcgan2 from:", new_repo_root)

from uvcgan2               import ROOT_OUTDIR, train
from uvcgan2.presets       import GEN_PRESETS, BH_PRESETS
from uvcgan2.utils.parsers import add_preset_name_parser, add_batch_size_parser

# ✅ ADD: Import custom adjacent pair dataset
from uvcgan2.data.adjacent_pair_dataset import AdjacentZPairDataset
from torchvision.transforms.functional import to_pil_image


def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description = '20260113_Inverted_Combined_BIT2HE_normal_duodenum_only_crypts_Train_3DFlow'
    )

    add_preset_name_parser(parser, 'gen',  GEN_PRESETS, 'uvcgan2')
    add_preset_name_parser(parser, 'head', BH_PRESETS,  'bn', 'batch head')

    parser.add_argument(
        '--no-pretrain', dest = 'no_pretrain', action = 'store_true',
        help = 'disable usage of the pre-trained generator'
    )

    parser.add_argument(
        '--lambda-gp', dest = 'lambda_gp', type = float,
        default = 0.01, help = 'magnitude of the gradient penalty'
    )

    parser.add_argument(
        '--lambda-cycle', dest = 'lambda_cyc', type = float,
        default = 10.0, help = 'magnitude of the cycle-consistency loss'
    )

    parser.add_argument(
        '--lr-gen', dest = 'lr_gen', type = float,
        default = 5e-5, help = 'learning rate of the generator'
    )
    
    parser.add_argument(
        '--root_data_path',
        type=str,
        required=True,
        help='Root path where train/test folders are located'
    )

    # ✅ Optional: expose z_spacing as CLI arg if you want
    parser.add_argument(
        '--z-spacing',
        type=int,
        default=2,
        help='Z-spacing for AdjacentZPairDataset (domain A only)'
    )

    parser.add_argument(
        '--lambda-sub-loss',
        type=float,
        default=0.0,
        help='Weight for the subtraction loss between adjacent slices in domain A'
    )

    parser.add_argument(
        '--lambda-embedding-loss',
        type=float,
        default=1.0,
        help='Weight for the embedding loss between adjacent slices in domain A'
    )

    add_batch_size_parser(parser, default = 1)

    return parser.parse_args()

def get_transfer_preset(cmdargs):
    if cmdargs.no_pretrain:
        return None

    base_model = (
        '/home/durrlab-asong/Anthony/UVCGANv2_vHE/outdir/20251213_Inverted_Combined_BIT2HE_normal_duodenum_only_crypts_Pretrain/'
        'model_m(autoencoder)_d(None)'
        f"_g({GEN_PRESETS[cmdargs.gen]['model']})_pretrain-{cmdargs.gen}"
    )

    return {
        'base_model' : base_model,
        'transfer_map'  : {
            'gen_ab' : 'encoder',
            'gen_ba' : 'encoder',
        },
        'strict'        : True,
        'allow_partial' : False,
        'fuzzy'         : None,
    }

cmdargs   = parse_cmdargs()
# /home/durrlab-asong/Anthony/subset_training_data_crypts
data_path_domainA = os.path.join(cmdargs.root_data_path, 'BIT', 'trainA')
data_path_domainB = os.path.join(cmdargs.root_data_path, 'FFPE_HE')

model_save_dir = os.path.join(ROOT_OUTDIR, '20260113_Inverted_Combined_BIT2HE_normal_duodenum_only_crypts_Train_3DFlow')
lambda_sub_str = str(cmdargs.lambda_sub_loss).replace('.', 'p')
lambda_emb_str = str(cmdargs.lambda_embedding_loss).replace('.', 'p')

# ✅ BUILD dataset config — domain A will use the AdjacentZPairDataset manually injected below
dataset_config = [
    {
        'dataset': {
            'name': 'adjacent-z-pairs',  # just a label; it will be overridden in train() pipeline
            'domain': 'A',
            'path': data_path_domainA,
            'z_spacing': cmdargs.z_spacing,  # pass to constructor
            'debug_root': os.path.join(cmdargs.root_data_path, 'debug_images')  # Optional: directory to save debug images from subtraction loss
        },
        'shape': (3, 512, 512),
        'transform_train': None,
        'transform_test': None,
    },
    {
        'dataset': {
            'name': 'cyclegan',
            'domain': 'B',
            'path': data_path_domainB,
        },
        'shape': (3, 512, 512),
        'transform_train': [
            { 'name': 'resize',      'size': 512 },
            { 'name': 'random-crop', 'size': 512 },
            'random-flip-horizontal',
        ],
        'transform_test': None,
    }
]

args_dict = {
    'batch_size' : cmdargs.batch_size,
    'data' : {
        'datasets'   : dataset_config,
        'merge_type' : 'unpaired',
        'workers'    : 1,
    },
    'epochs'      : 200,
    'discriminator' : {
        'model'      : 'basic',
        'model_args' : { 'shrink_output' : False },
        'optimizer'  : {
            'name'  : 'Adam',
            'lr'    : 1e-4,
            'betas' : (0.5, 0.99),
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        },
        'spectr_norm' : True,
    },
    'generator' : {
        **GEN_PRESETS[cmdargs.gen],
        'optimizer'  : {
            'name'  : 'Adam',
            'lr'    : cmdargs.lr_gen,
            'betas' : (0.5, 0.99),
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        },
    },
    'model' : 'uvcgan2_3D_embedding_loss',
    'model_args' : {
        'lambda_a'        : cmdargs.lambda_cyc,
        'lambda_b'        : cmdargs.lambda_cyc,
        'lambda_idt'      : 0.5,
        'lambda_subtraction_loss' : cmdargs.lambda_sub_loss,  # You can adjust this weight as needed
        'lambda_embedding_loss' : cmdargs.lambda_embedding_loss,  # You can adjust this weight as needed
        'avg_momentum'    : 0.9999,
        'head_queue_size' : 3,
        'z_spacing' : cmdargs.z_spacing,  # Pass z_spacing to the main config for use in the model
        'debug_root': os.path.join(model_save_dir, f'debug_images_zspacing={cmdargs.z_spacing}_lambdsub={lambda_sub_str}_lambdemb={lambda_emb_str}'),  # Optional: directory to save debug images from subtraction loss
        'head_config'     : {
            'name'            : BH_PRESETS[cmdargs.head],
            'input_features'  : 512,
            'output_features' : 1,
            'activ'           : 'leakyrelu',
        },
    },
    'gradient_penalty' : {
        'center'    : 0,
        'lambda_gp' : cmdargs.lambda_gp,
        'mix_type'  : 'real-fake',
        'reduction' : 'mean',
    },
    'scheduler'       : None,
    'loss'            : 'lsgan',
    'steps_per_epoch' : 2000,
    'transfer'        : get_transfer_preset(cmdargs),

    # Training label for bookkeeping
    'label'  : (
        f'{cmdargs.gen}-{cmdargs.head}_({cmdargs.no_pretrain}'
        f':{cmdargs.lambda_cyc}:{cmdargs.lambda_gp}:{cmdargs.lr_gen})'
    ),

    'outdir'     : os.path.join(model_save_dir, f'20260106_Inverted_Combined_BIT2HE_normal_duodenum_only_crypts_Train_3DFlow_zspacing={cmdargs.z_spacing}slices_lambdsub={lambda_sub_str}_lambdemb={lambda_emb_str}'),
    'log_level'  : 'DEBUG',
    'checkpoint' : 10,
}
print(ROOT_OUTDIR)

### Debug the dataloader. 
# Inspect domain A dataset
datasetA = AdjacentZPairDataset(root_dir=data_path_domainA, z_spacing=cmdargs.z_spacing)
print("\n=== Sample Z-adjacent pairs from Domain A ===")
for i in range(min(10, len(datasetA))):
    item = datasetA[i]
    print(f"[{i}] z1: {item['z1_name']}  |  z2: {item['z2_name']}  |  Z: ({item['meta']['z_t']} → {item['meta']['z_t_plus']})  |  Patch: {item['meta']['patch_id']}")

# Save images from the first pair
model_save_dir = cmdargs.model_save_dir if hasattr(cmdargs, 'model_save_dir') else './debug_output'
os.makedirs(model_save_dir, exist_ok=True)

item0 = datasetA[0]
imgA = to_pil_image(item0['z1'])
imgB = to_pil_image(item0['z2'])

save_path_A = os.path.join(model_save_dir, f"debug_A_{item0['z1_name']}.png")
save_path_B = os.path.join(model_save_dir, f"debug_B_{item0['z2_name']}.png")   
imgA.save(save_path_A)
imgB.save(save_path_B)

print(f"Saved debug images to:\n{save_path_A}\n{save_path_B}")


# ✅ Final call
train(args_dict)
