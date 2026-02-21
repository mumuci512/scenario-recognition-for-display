import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os
import argparse
import sys
import torch.nn.functional as F

# --- Attention Modules ---

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class CBAM(nn.Module):
    def __init__(self, in_planes, learnable=False, init_weight=0.2):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()
        
        # Parameter logic for learnable center weight
        if learnable:
            val = min(max(init_weight, 0.0), 0.5)
            self.center_bias = nn.Parameter(torch.tensor([val]))
        else:
            self.register_buffer('center_bias', torch.tensor([init_weight]))

    def forward(self, x):
        x = x * self.ca(x)
        sa_map = self.sa(x)
        # Detach feature for multiplication, keep map for loss
        return x * sa_map.detach(), sa_map

class GuidedModel(nn.Module):
    def __init__(self, model_name, num_classes, learnable=False, init_weight=0.2):
        super(GuidedModel, self).__init__()
        self.model_name = model_name
        
        if model_name == 'resnet18':
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.features = nn.Sequential(*list(base.children())[:-2])
            self.cbam = CBAM(512, learnable, init_weight)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512, num_classes)
        elif model_name == 'mobilenet_v2':
            base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            self.features = base.features
            self.cbam = CBAM(1280, learnable, init_weight)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(1280, num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def forward(self, x):
        x = self.features(x)
        x, att = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        if self.model_name == 'resnet18':
            x = self.fc(x)
        else:
            x = self.classifier(x)
        return x, att

# --- Mask Generators ---

def generate_hard_mask(batch_size, device, size=7):
    """Original 'Hard' Mask: 1 on borders, 0 in center."""
    mask = torch.zeros((size, size), device=device)
    mask[0:2, :] = 1
    mask[-2:, :] = 1
    mask[:, 0:2] = 1
    mask[:, -2:] = 1
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, size, size)

def generate_smooth_mask(batch_size, device, size=7, center_weight=0.2):
    """New 'Smooth' Mask: Gaussian blur + variable center weight."""
    mask = torch.ones((size, size), device=device)
    inner_start, inner_end = 2, size - 2
    
    if inner_end > inner_start:
        mask[inner_start:inner_end, inner_start:inner_end] = center_weight
        
        # Apply 3x3 Gaussian blur
        mask = mask.view(1, 1, size, size)
        kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], 
                             dtype=torch.float32, device=device)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, 3, 3)
        
        mask = F.conv2d(mask, kernel, padding=1)
        mask = mask.view(size, size)

    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, 1, 1)

# --- Utilities ---

def get_dataloaders(data_dir, batch_size, val_split=0.2):
    # Training transforms (Agumentation)
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Validation transforms
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(data_dir, transform=train_tf)
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    val_ds.dataset.transform = val_tf
    
    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    }
    return loaders, dataset.classes, dataset.class_to_idx

# --- Training Loop ---

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loaders, class_names, class_to_idx = get_dataloaders(args.data_dir, args.batch_size)
    
    # Target classes for guidance
    target_games = [
        'CSGO', 'CF', 'Overwatch', 'DeltaForce', 'PUBG', 'Apex',
        'LOL', 'Dota2', 'Genshin', 'Kart', 'Forza', 'FWJ', 'QQSpeed', 'yanyun'
    ]
    guided_indices = set()
    for name in class_names:
        if any(g.lower() in name.lower() for g in target_games):
            guided_indices.add(class_to_idx[name])

    # Model Setup
    model = GuidedModel(
        args.model, 
        len(class_names), 
        learnable=args.learnable, 
        init_weight=args.init_weight
    ).to(device)
    
    # Optimizer (Lower LR for CBAM)
    optimizer = optim.SGD([
        {'params': [p for n, p in model.named_parameters() if 'cbam' not in n], 'lr': args.lr},
        {'params': model.cbam.parameters(), 'lr': args.lr * 0.1}
    ], momentum=0.9)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_attn = nn.MSELoss()

    ckpt_dir = f'checkpoints_{args.model}'
    os.makedirs(ckpt_dir, exist_ok=True)
    
    best_acc = 0.0
    start_epoch = 0

    if args.resume:
        path = os.path.join(ckpt_dir, 'best.pth')
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            best_acc = ckpt.get('best_acc', 0.0)
            start_epoch = ckpt['epoch']
            print(f"Resumed from epoch {start_epoch}")

    print(f"Config: Mask={args.mask_type}, Learnable={args.learnable}")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Log weight status if learnable
        if args.learnable:
            curr_w = model.cbam.center_bias.item()
            print(f"Current Center Weight: {curr_w:.4f}")

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            
            running_loss = 0.0
            corrects = 0
            total = 0
            
            for inputs, labels in loaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, att_map = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    loss = criterion_cls(outputs, labels)
                    
                    # Attention Guidance Loss
                    is_target = torch.tensor([l.item() in guided_indices for l in labels], device=device)
                    
                    if is_target.any():
                        target_att = att_map[is_target]
                        size = target_att.size(2)
                        
                        # Generate Mask based on Type
                        if args.mask_type == 'hard':
                            mask = generate_hard_mask(target_att.size(0), device, size)
                        else:
                            # Clamp weight for mask generation just in case
                            w = torch.clamp(model.cbam.center_bias, 0.0, 0.5)
                            mask = generate_smooth_mask(target_att.size(0), device, size, w)
                            
                        loss += args.lambda_attn * criterion_attn(target_att, mask)

                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        
                        # Constraint: clamp learnable parameter [0.0, 0.5]
                        if args.learnable:
                            with torch.no_grad():
                                model.cbam.center_bias.clamp_(0.0, 0.5)
                
                running_loss += loss.item() * inputs.size(0)
                corrects += torch.sum(preds == labels.data)
                total += inputs.size(0)
            
            epoch_loss = running_loss / total
            epoch_acc = corrects.double() / total
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Save Checkpoint
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'center_bias': model.cbam.center_bias.item()
                }, os.path.join(ckpt_dir, 'best.pth'))
                print(f"New best model: {best_acc:.4f}")

    print("Training complete.")

def parse_args():
    parser = argparse.ArgumentParser(description='Unified Guided CBAM Training')
    parser.add_argument('--data_dir', default='dataset', help='Dataset directory')
    parser.add_argument('--model', default='mobilenet_v2', choices=['resnet18', 'mobilenet_v2'])
    
    # Mask Configuration
    parser.add_argument('--mask_type', default='hard', choices=['hard', 'smooth'], 
                        help="hard: 0-center/1-edge; smooth: gaussian blurred center")
    parser.add_argument('--learnable', action='store_true', 
                        help="Make center weight learnable (only for smooth mask)")
    parser.add_argument('--init_weight', type=float, default=0.2, 
                        help="Initial center weight for smooth mask (default 0.2)")
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lambda_attn', type=float, default=1.0)
    parser.add_argument('--resume', action='store_true')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.learnable and args.mask_type == 'hard':
        print("Warning: --learnable is ignored when mask_type is 'hard'.")
        
    train(args)