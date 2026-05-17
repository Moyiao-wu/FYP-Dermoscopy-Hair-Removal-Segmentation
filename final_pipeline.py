import os, sys, cv2, torch, torch.nn as nn, numpy as np
from omegaconf import OmegaConf

# --- 1. SET PATHS  ---
ROOT_DIR = "/gpfs/work/bio/yixuanli2204/FYP_Project"
sys.path.append(os.path.join(ROOT_DIR, "code/lama"))
sys.path.append(os.path.join(ROOT_DIR, "code/UltraLight-VM-UNet"))

from saicinpainting.training.trainers import load_checkpoint as load_lama
from saicinpainting.evaluation.utils import move_to_device
from models.UltraLight_VM_UNet import UltraLight_VM_UNet
from segment_anything import sam_model_registry

# --- 2. DEFINE MODEL CLASSES  ---
class SamAdapter(nn.Module):
    def __init__(self, dim=256, adapter_dim=64):
        super().__init__()
        self.adapter = nn.Sequential(nn.Linear(dim, adapter_dim), nn.GELU(), nn.Linear(adapter_dim, dim))
        nn.init.zeros_(self.adapter[-1].weight); nn.init.zeros_(self.adapter[-1].bias)
    def forward(self, x): return x + self.adapter(x)

class SAMwithAdapter(nn.Module):
    def __init__(self, model_type="vit_h", checkpoint=None):
        super().__init__()
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        for param in self.sam.parameters(): param.requires_grad = False
        self.adapter = SamAdapter(dim=256)
        self.mask_head = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))
    def forward(self, image, diff_map):
        feat_img = self.sam.image_encoder(image)
        with torch.no_grad(): feat_diff = self.sam.image_encoder(diff_map)
        features = feat_img + feat_diff 
        b, c, h, w = features.shape
        x = self.adapter(features.permute(0, 2, 3, 1).reshape(b, h*w, c))
        features = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        logits = self.mask_head(features)
        return torch.sigmoid(nn.functional.interpolate(logits, size=(1024, 1024), mode='bilinear'))

# --- 3. LOAD ALL MODELS ONCE ---
device = torch.device("cuda")

# Stage 1 Weights
sam_model = SAMwithAdapter("vit_h", f"{ROOT_DIR}/weights/sam/sam_vit_h_4b8939.pth").to(device)
sam_model.load_state_dict(torch.load(f"{ROOT_DIR}/weights/sam/sam_adapter_vith_best.pth"))
sam_model.eval()

# Stage 2 Weights
lama_cfg = OmegaConf.load(f"{ROOT_DIR}/weights/lama/config.yaml")
lama_cfg.training_model.predict_only = True
lama_model = load_lama(lama_cfg, f"{ROOT_DIR}/weights/lama/bigLama.ckpt", strict=False, map_location='cpu')
lama_model.to(device).eval()

# Stage 3 Weights
vmunet_model = UltraLight_VM_UNet(num_classes=1, c_list=[8,16,24,32,48,64], bridge=True).to(device)
vmunet_model.load_state_dict(torch.load(f"{ROOT_DIR}/weights/vmunet/vmunet_finetuned_fixed_seed.pth"))
vmunet_model.eval()

# --- 4. THE PIPELINE  ---
def run_exact_pipeline(img_id):
    img_path = f"{ROOT_DIR}/data/gold_real_d2/dermoscopic_image/{img_id}.png"
    ovl_path = f"{ROOT_DIR}/data/gold_real_d2/overlay/{img_id}.png"
    
    # 0. Read Image
    img_bgr = cv2.imread(img_path)
    ovl_bgr = cv2.imread(ovl_path)
    diff_bgr = cv2.absdiff(img_bgr, ovl_bgr)
    h_orig, w_orig = img_bgr.shape[:2]

    # STAGE 1: SAM 
    def prep_sam(x):
        tmp = cv2.resize(cv2.cvtColor(x, cv2.COLOR_BGR2RGB), (1024, 1024))
        return torch.from_numpy(tmp).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0
    
    with torch.no_grad():
        mask_t = sam_model(prep_sam(img_bgr), prep_sam(diff_bgr))
        hair_mask_raw = (mask_t.squeeze().cpu().numpy() * 255).astype(np.uint8)
        _, hair_mask_bin = cv2.threshold(hair_mask_raw, 127, 255, cv2.THRESH_BINARY)

    # STAGE 2: LAMA 
    img_res = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (512, 512))
    mask_res = cv2.resize(hair_mask_bin, (512, 512))
    _, mask_bin_512 = cv2.threshold(mask_res, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask_final_512 = cv2.dilate(mask_bin_512, kernel, iterations=1)

    batch = {
        "image": torch.from_numpy(img_res).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0,
        "mask": torch.from_numpy(mask_final_512).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0
    }
    with torch.no_grad():
        output_lama = lama_model(batch)
        res_lama = output_lama["inpainted"][0].permute(1, 2, 0).cpu().numpy()
        res_lama = (np.clip(res_lama, 0, 1) * 255).astype(np.uint8)
        # Convert back to BGR for VM-UNet and Saving
        res_lama_bgr = cv2.resize(cv2.cvtColor(res_lama, cv2.COLOR_RGB2BGR), (w_orig, h_orig))

    # STAGE 3: VM-UNET 
    img_256 = cv2.resize(cv2.cvtColor(res_lama_bgr, cv2.COLOR_BGR2RGB), (256, 256))
    img_v_t = torch.from_numpy(img_256).permute(2,0,1).unsqueeze(0).float().to(device) / 255.0
    with torch.no_grad():
        pred_t = vmunet_model(img_v_t)
        lesion_mask = (pred_t.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

    # --- VISUALIZATION (512 grid) ---
    sz = (512, 512)
    v1 = cv2.resize(img_bgr, sz)
    v2 = cv2.resize(cv2.cvtColor(hair_mask_bin, cv2.COLOR_GRAY2BGR), sz)
    v3 = cv2.resize(res_lama_bgr, sz)
    v4 = cv2.resize(cv2.cvtColor(lesion_mask, cv2.COLOR_GRAY2BGR), sz)
    
    final_canvas = np.vstack([np.hstack([v1, v2]), np.hstack([v3, v4])])
    
    # Save the result
    cv2.imwrite(f"INTEGRATED_RESULT_{img_id}.png", final_canvas)
    print(f"✅ Success! Generated INTEGRATED_RESULT_{img_id}.png")

# Run test on one image
run_exact_pipeline("ISIC_0010025")