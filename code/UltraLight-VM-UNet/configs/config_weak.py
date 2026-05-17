from torchvision import transforms
from utils import *
from datetime import datetime

class setting_config:

    network = 'UltraLight_VM_UNet' 
    model_config = {
        'num_classes': 1,
        'input_channels': 3,
        'c_list': [8,16,24,32,48,64],
        'split_att': 'fc',
        'bridge': True,
    }

    test_weights = ''
    datasets = 'ISIC_Combined'
    data_path = '/gpfs/work/bio/yixuanli2204/FYP_Project/code/UltraLight-VM-UNet/data/skin_lesion/'

    criterion = BceDiceLoss()

    num_classes = 1
    input_size_h = 256
    input_size_w = 256
    input_channels = 3
    distributed = False
    local_rank = -1
    num_workers = 8
    seed = 42
    amp = False
    batch_size = 8
    epochs = 50

    work_dir = 'results/VMUNET_WEAK_BASELINE/'

    print_interval = 20
    val_interval = 2
    save_interval = 10
    threshold = 0.5

    
    opt = 'AdamW'
    lr = 0.001
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 1e-2
    amsgrad = False

    
    sch = 'CosineAnnealingLR'
    T_max = 50
    eta_min = 0.00001
    last_epoch = -1 
    
    
    step_size = 30
    gamma = 0.1
    milestones = [60, 120, 150]
    mode = 'min'
    factor = 0.1
    patience = 10
    cooldown = 0
    min_lr = 0