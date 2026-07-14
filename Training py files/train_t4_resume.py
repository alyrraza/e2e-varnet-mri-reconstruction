import sys
sys.path.insert(0, '/workspace/data/fastMRI')

import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import fastmri
from fastmri.data import subsample, transforms as T
from fastmri.data.mri_data import SliceDataset
from fastmri.models import VarNet
from fastmri.losses import SSIMLoss

NUM_CASCADES = 4
CHANS = 18
SENS_CHANS = 8
LR = 3e-4
TOTAL_EPOCHS = 50
GRAD_ACCUM_STEPS = 8
PRINT_EVERY_SECONDS = 210

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}", flush=True)

mask_func = subsample.EquispacedMaskFractionFunc(center_fractions=[0.08], accelerations=[4])
train_transform = T.VarNetDataTransform(mask_func=mask_func)
val_transform = T.VarNetDataTransform(mask_func=mask_func)

train_dataset = SliceDataset(root="/workspace/train_combined", transform=train_transform, challenge="singlecoil")
val_dataset = SliceDataset(root="/workspace/val_combined", transform=val_transform, challenge="singlecoil")

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=12, persistent_workers=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=12, persistent_workers=False)

model = VarNet(num_cascades=NUM_CASCADES, chans=CHANS, sens_chans=SENS_CHANS).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = SSIMLoss().to(device)

checkpoint = torch.load("/workspace/data/checkpoints_backup/checkpoint_epoch_10.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1

print(f"Resumed from checkpoint. Last completed epoch (0-indexed): {checkpoint['epoch']}", flush=True)
print(f"Resuming training from epoch: {start_epoch + 1} (1-indexed)", flush=True)
print(f"Previous val SSIM: {checkpoint.get('val_ssim', 'N/A')}\n", flush=True)

best_checkpoint = torch.load("/workspace/data/checkpoints_backup/best_model.pt", map_location=device)
best_val_ssim = best_checkpoint['val_ssim']
print(f"Current best val SSIM so far: {best_val_ssim:.4f} (epoch {best_checkpoint['epoch']+1})\n", flush=True)

baseline_ssim = 0.7453

last_print_time = time.time()

for epoch in range(start_epoch, TOTAL_EPOCHS):
    model.train()
    epoch_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS}", leave=True)

    for i, batch in enumerate(pbar):
        masked_kspace, mask, num_low_freqs, target, fname, slice_num, max_value, crop_size = batch
        masked_kspace = masked_kspace.unsqueeze(1).to(device)
        mask = mask.unsqueeze(1).to(device)
        target = target.to(device)
        max_value = max_value.to(device)

        output = model(masked_kspace, mask)
        output_cropped = T.center_crop(output, target.shape[-2:])
        loss = loss_fn(output_cropped.unsqueeze(1), target.unsqueeze(1), data_range=max_value)
        loss = loss / GRAD_ACCUM_STEPS
        loss.backward()

        if (i + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item() * GRAD_ACCUM_STEPS
        pbar.set_postfix({"loss": f"{loss.item()*GRAD_ACCUM_STEPS:.4f}"})

        current_time = time.time()
        if current_time - last_print_time >= PRINT_EVERY_SECONDS:
            print(f"[STATUS] Epoch {epoch+1}/{TOTAL_EPOCHS}, Batch {i+1}/{len(train_loader)}, Running avg loss: {epoch_loss/(i+1):.4f}", flush=True)
            last_print_time = current_time

    avg_epoch_loss = epoch_loss / len(train_loader)

    model.eval()
    val_ssim_total = 0.0
    with torch.no_grad():
        for batch in val_loader:
            masked_kspace, mask, num_low_freqs, target, fname, slice_num, max_value, crop_size = batch
            masked_kspace = masked_kspace.unsqueeze(1).to(device)
            mask = mask.unsqueeze(1).to(device)
            target = target.to(device)
            max_value = max_value.to(device)
            output = model(masked_kspace, mask)
            output_cropped = T.center_crop(output, target.shape[-2:])
            ssim_val = 1 - loss_fn(output_cropped.unsqueeze(1), target.unsqueeze(1), data_range=max_value).item()
            val_ssim_total += ssim_val
    avg_val_ssim = val_ssim_total / len(val_loader)

    print(f"\n=== Epoch {epoch+1} complete ===", flush=True)
    print(f"Train loss (1-SSIM): {avg_epoch_loss:.4f}", flush=True)
    print(f"Val SSIM: {avg_val_ssim:.4f}  (zero-filled baseline: {baseline_ssim:.4f})", flush=True)
    print(f"Improvement over baseline: {avg_val_ssim - baseline_ssim:+.4f}\n", flush=True)

    torch.save({
        'epoch': epoch, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_epoch_loss, 'val_ssim': avg_val_ssim,
    }, f"/workspace/checkpoint_epoch_{epoch+1}.pt")

    if avg_val_ssim > best_val_ssim:
        best_val_ssim = avg_val_ssim
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_ssim': avg_val_ssim}, "/workspace/best_model.pt")
        print(f"[NEW BEST] Saved best_model.pt (val SSIM: {avg_val_ssim:.4f})\n", flush=True)

print("=== TRAINING COMPLETE ===", flush=True)
print(f"Best val SSIM achieved: {best_val_ssim:.4f}", flush=True)
