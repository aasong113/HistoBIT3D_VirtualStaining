import argparse
import getpass
import sys
import os
from datetime import date

# Go up 3 levels to get repo root from inside scripts/2025*/train_3D.py
new_repo_root = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
# Remove old UVCGAN paths and add the correct one
sys.path = [p for p in sys.path if 'UVCGANv2_vHE' not in p or new_repo_root in os.path.abspath(p)]
sys.path.insert(0, new_repo_root)
print("Using uvcgan2 from:", new_repo_root)

os.environ["WANDB_API_KEY"] = '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'

from uvcgan2               import ROOT_OUTDIR, train
from uvcgan2.presets       import GEN_PRESETS, BH_PRESETS
from uvcgan2.utils.parsers import add_preset_name_parser, add_batch_size_parser

# ✅ ADD: Import custom adjacent pair dataset
from uvcgan2.data.adjacent_pair_dataset import AdjacentZPairDataset
from torchvision.transforms.functional import to_pil_image

today_str = date.today().strftime('%Y%m%d')

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description = f'{today_str}_BIT2HE_normal_duodenum_only_crypts_3DFlow'
    )

    add_preset_name_parser(parser, 'gen',  GEN_PRESETS, 'uvcgan2')
    add_preset_name_parser(parser, 'head', BH_PRESETS,  'bn', 'batch head')

    parser.add_argument(
        '--no-pretrain', dest = 'no_pretrain', action = 'store_true',
        help = 'disable usage of the pre-trained generator'
    )
    parser.add_argument(
        '--base-model',
        dest='base_model',
        type=str,
        default=None,
        help='Path to a pretrained model directory (overrides the default pretrain path)'
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

    parser.add_argument(
        '--lambda-style-fusion',
        type=float,
        default=1.0,
        help='Initial scale for style-token injection (cosine decays over training epochs)'
    )

    parser.add_argument(
        '--lambda-style-loss',
        type=float,
        default=1.0,
        help='Weight for ViT bottleneck style-stat loss (gen_ba(real_b) vs gen_ba(fake_b))'
    )

    parser.add_argument(
        '--style-fusion-inject',
        choices=['add', 'adain'],
        default='adain',
        help="How to inject the style delta into the A->B ViT style token: 'add' or 'adain'"
    )

    parser.add_argument(
        '--use-style-fusion',
        dest='use_style_fusion',
        action='store_true',
        help='Enable style fusion (uses --lambda-style-fusion weight)'
    )
    parser.add_argument(
        '--no-style-fusion',
        dest='use_style_fusion',
        action='store_false',
        help='Disable style fusion regardless of weight'
    )
    # Backwards compatible aliases (misnamed historically).
    parser.add_argument(
        '--use-embedding-loss',
        dest='use_style_fusion',
        action='store_true',
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--no-embedding-loss',
        dest='use_style_fusion',
        action='store_false',
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(use_style_fusion=True)

    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    parser.add_argument(
        '--wandb-api-key',
        type=str,
        nargs='?',
        const='__PROMPT__',
        default=None,
        help='W&B API key; pass a value or use `--wandb-api-key` alone to be prompted (prefer WANDB_API_KEY env var or `wandb login`)'
    )
    parser.add_argument(
        '--wandb-api-key-file',
        type=str,
        default=None,
        help='Path to a file containing the W&B API key (recommended over --wandb-api-key)'
    )
    parser.add_argument(
        '--wandb-entity',
        type=str,
        default='sanhong113-johns-hopkins-university',
        help='wandb entity/team'
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default=None,
        help='wandb project name (defaults to an auto-generated name)'
    )
    parser.add_argument(
        '--wandb-mode',
        choices=['online', 'offline', 'disabled'],
        default='online',
        help="wandb mode; use 'offline' on airgapped machines"
    )

    add_batch_size_parser(parser, default = 1)

    return parser.parse_args()

def get_transfer_preset(cmdargs):
    if cmdargs.no_pretrain:
        return None

    if cmdargs.base_model is not None:
        base_model = cmdargs.base_model
    else:
        pretrain_root = (
            '/home/durrlab/Desktop/Anthony/UGVSM/UVCGANv2_vHE/outdir/'
            '20251225_Inverted_combined_BIT2HE_duodenum_crypts/'
            '20251225_Inverted_combined_BIT2HE_duodenum_crypts_pretrain'
        )
        model_dir = (
            'model_m(autoencoder)_d(None)'
            f"_g({GEN_PRESETS[cmdargs.gen]['model']})_pretrain-{cmdargs.gen}"
        )
        base_model = os.path.join(pretrain_root, model_dir)

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
if not cmdargs.use_style_fusion:
    cmdargs.lambda_style_fusion = 0.0
# /home/durrlab-asong/Anthony/subset_training_data_crypts
data_path_domainA = os.path.join(cmdargs.root_data_path, 'BIT', 'trainA')
data_path_domainB = os.path.join(cmdargs.root_data_path, 'FFPE_HE')

model_save_dir = os.path.join(ROOT_OUTDIR, f'{today_str}_BIT2HE_normal_duodenum_only_crypts_3DFlow')
lambda_sub_str = str(cmdargs.lambda_sub_loss).replace('.', 'p')
lambda_emb_str = str(cmdargs.lambda_embedding_loss).replace('.', 'p')
lambda_sty_str = str(cmdargs.lambda_style_fusion).replace('.', 'p')
wandb_project = cmdargs.wandb_project or (
    f'{today_str}_duodenum_only_crypts_3DFlow_'
    f'zspacing={cmdargs.z_spacing}slices_'
    f'lamsub={lambda_sub_str}_lamemb={lambda_emb_str}_lamSty={lambda_sty_str}'
)

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
    'model' : 'uvcgan2_3D_stylefusion',
    'model_args' : {
        'lambda_a'        : cmdargs.lambda_cyc,
        'lambda_b'        : cmdargs.lambda_cyc,
        'lambda_idt'      : 0.5,
        'lambda_subtraction_loss' : cmdargs.lambda_sub_loss,  # You can adjust this weight as needed
        'lambda_embedding_loss' : cmdargs.lambda_embedding_loss,  # You can adjust this weight as needed
        'lambda_style_loss' : cmdargs.lambda_style_loss,
        'lambda_style_fusion' : cmdargs.lambda_style_fusion,
        'style_fusion_inject' : cmdargs.style_fusion_inject,
        'avg_momentum'    : 0.9999,
        'head_queue_size' : 3,
        'z_spacing' : cmdargs.z_spacing,  # Pass z_spacing to the main config for use in the model
        'debug_root': os.path.join(model_save_dir, f'debug_images_zspacing={cmdargs.z_spacing}_lambdsub={lambda_sub_str}_lambdemb={lambda_emb_str}_lamSty={lambda_sty_str}'),  # Optional: directory to save debug images from subtraction loss
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

    'outdir'     : os.path.join(model_save_dir, f'{today_str}_duodenum_crypts_3DFlow_style_fusion_zspacing={cmdargs.z_spacing}slices_lamsub={lambda_sub_str}_lamemb={lambda_emb_str}_lamSty={lambda_sty_str}'),
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

if cmdargs.wandb and cmdargs.wandb_mode != 'disabled':
    # Keep wandb optional; training should still run if wandb is not installed.
    try:
        import wandb  # type: ignore
    except Exception as e:
        print(f"[wandb] Disabled (import failed): {e}")
        wandb = None
    else:
        if cmdargs.wandb_api_key_file:
            try:
                with open(cmdargs.wandb_api_key_file, "r", encoding="utf-8") as f:
                    os.environ["WANDB_API_KEY"] = f.read().strip()
            except Exception as e:
                print(f"[wandb] Disabled (failed to read --wandb-api-key-file): {e}")
                wandb = None
        if wandb is not None and cmdargs.wandb_api_key == '__PROMPT__':
            if not sys.stdin.isatty():
                print("[wandb] Disabled (--wandb-api-key provided without a value in a non-interactive session)")
                wandb = None
            else:
                os.environ["WANDB_API_KEY"] = getpass.getpass("W&B API key: ").strip()
        if wandb is not None and cmdargs.wandb_api_key:
            os.environ["WANDB_API_KEY"] = cmdargs.wandb_api_key.strip()
        os.environ['WANDB_MODE'] = cmdargs.wandb_mode
        try:
            if os.environ.get("WANDB_API_KEY"):
                try:
                    wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
                except Exception as e:
                    print(f"[wandb] Disabled (login failed): {e}")
                    raise
            wandb.init(
                entity = cmdargs.wandb_entity,
                project = wandb_project,
                config = {
                    'data_path_domainA'     : data_path_domainA,
                    'data_path_domainB'     : data_path_domainB,
                    'lambda_sub_str'        : lambda_sub_str,
                    'lambda_emb_str'        : lambda_emb_str,
                    'lambda_sty_str'        : lambda_sty_str,
                    'z_spacing'             : cmdargs.z_spacing,
                    'lambda_sub_loss'       : cmdargs.lambda_sub_loss,
                    'lambda_embedding_loss' : cmdargs.lambda_embedding_loss,
                    'lambda_style_fusion'   : cmdargs.lambda_style_fusion,
                    'style_fusion_inject'   : cmdargs.style_fusion_inject,
                    'use_style_fusion'      : cmdargs.use_style_fusion,
                    'epochs'                : args_dict['epochs'],
                    'lr_gen'                : cmdargs.lr_gen,
                },
            )
        except Exception as e:
            print(f"[wandb] Disabled (init failed): {e}")
            wandb = None


# ✅ Final call
train(args_dict)

try:
    import wandb  # type: ignore
except Exception:
    wandb = None
if wandb is not None and getattr(wandb, "run", None) is not None:
    wandb.finish()
