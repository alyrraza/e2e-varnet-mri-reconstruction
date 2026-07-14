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

NUM_CASCADES = 6
CHANS = 18
SENS_CHANS = 8
LR = 3e-4
TOTAL_EPOCHS = 50   # pehla epoch dekh ke adjust karenge zaroorat pade to
GRAD_ACCUM_STEPS = 8
PRINT_EVERY_SECONDS = 210

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}", flush=True)

mask_func = subsample.EquispacedMaskFractionFunc(center_fractions=[0.08], accelerations=[4])
train_transform = T.VarNetDataTransform(mask_func=mask_func)
val_transform = T.VarNetDataTransform(mask_func=mask_func)

train_dataset = SliceDataset(root="/workspace/train_combined", transform=train_transform, challenge="singlecoil")
val_dataset = SliceDataset(root="/workspace/val_combined", transform=val_transform, challenge="singlecoil")

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8, persistent_workers=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, persistent_workers=False)

model = VarNet(num_cascades=NUM_CASCADES, chans=CHANS, sens_chans=SENS_CHANS).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = SSIMLoss().to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"T=6 model parameters: {total_params:,}", flush=True)

baseline_ssim = 0.7453
best_val_ssim = -1.0
last_print_time = time.time()
epoch_times = []

for epoch in range(TOTAL_EPOCHS):
    model.train()
    epoch_loss = 0.0
    optimizer.zero_grad()
    epoch_start = time.time()

    pbar = tqdm(train_loader, desc=f"T=6 Epoch {epoch+1}/{TOTAL_EPOCHS}", leave=True)

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
            print(f"[STATUS] T=6 Epoch {epoch+1}/{TOTAL_EPOCHS}, Batch {i+1}/{len(train_loader)}, Running avg loss: {epoch_loss/(i+1):.4f}", flush=True)
            last_print_time = current_time

    avg_epoch_loss = epoch_loss / len(train_loader)
    epoch_duration = time.time() - epoch_start
    epoch_times.append(epoch_duration)

    model.eval()
    val_ssim_total = 0.0
    val_count = 0
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
            val_count += 1
    avg_val_ssim = val_ssim_total / val_count

    print(f"\n=== T=6 Epoch {epoch+1} complete (took {epoch_duration/60:.1f} min) ===", flush=True)
    print(f"Train loss (1-SSIM): {avg_epoch_loss:.4f}", flush=True)
    print(f"Val SSIM: {avg_val_ssim:.4f}  (zero-filled baseline: {baseline_ssim:.4f})", flush=True)
    print(f"Improvement over baseline: {avg_val_ssim - baseline_ssim:+.4f}", flush=True)

    # Budget projection — har epoch ke baad update hota hai taake pata chale kitna waqt aur lagega
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    remaining_epochs = TOTAL_EPOCHS - (epoch + 1)
    est_remaining_hours = (avg_epoch_time * remaining_epochs) / 3600
    print(f"[BUDGET] Avg epoch time so far: {avg_epoch_time/60:.1f} min. Est. remaining time for {remaining_epochs} more epochs: {est_remaining_hours:.2f} hours\n", flush=True)

    torch.save({
        'epoch': epoch, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_epoch_loss, 'val_ssim': avg_val_ssim, 'num_cascades': NUM_CASCADES,
    }, f"/workspace/t6_checkpoint_epoch_{epoch+1}.pt")

    if avg_val_ssim > best_val_ssim:
        best_val_ssim = avg_val_ssim
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_ssim': avg_val_ssim, 'num_cascades': NUM_CASCADES}, "/workspace/t6_best_model.pt")
        print(f"[NEW BEST] Saved t6_best_model.pt (val SSIM: {avg_val_ssim:.4f})\n", flush=True)

print("=== T=6 TRAINING COMPLETE ===", flush=True)
print(f"Best val SSIM achieved: {best_val_ssim:.4f}", flush=True)
