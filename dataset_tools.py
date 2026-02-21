import os
import sys
import shutil
import argparse
import numpy as np
import imagehash
from PIL import Image

# 支持的图片格式
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

def get_image_files(root_dir):
    """生成器：递归获取所有图片文件路径"""
    for root, _, files in os.walk(root_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in VALID_EXTS:
                yield os.path.join(root, file)

def cmd_trim(args):
    """去除黑边模式"""
    print(f"Mode: Trim Black Borders | Threshold: {args.threshold}")
    print(f"Path: {args.path}")
    
    count_scanned = 0
    count_cropped = 0

    for file_path in get_image_files(args.path):
        count_scanned += 1
        try:
            with Image.open(file_path) as img:
                arr = np.array(img.convert('RGB'))
                # 计算是否超过阈值
                mask = np.mean(arr, axis=2) > args.threshold

                if not np.any(mask):
                    continue

                coords = np.argwhere(mask)
                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0) + 1

                # 只有当裁剪框小于原图时才保存
                if (x1 - x0) < img.width or (y1 - y0) < img.height:
                    cropped = img.crop((x0, y0, x1, y1))
                    cropped.save(file_path, quality=95)
                    count_cropped += 1
                    sys.stdout.write(f"\rCropped: {count_cropped} / Scanned: {count_scanned}")
                    sys.stdout.flush()

        except Exception:
            continue

    print(f"\nDone. Scanned: {count_scanned}, Cropped: {count_cropped}")

def cmd_filter(args):
    """筛选非横屏图片模式 (宽 <= 高)"""
    trash_dir = os.path.join(args.path, "_excluded_portrait")
    print(f"Mode: Filter Portrait Images (Width <= Height)")
    print(f"Path: {args.path} -> Moving excluded to: {trash_dir}")

    count_scanned = 0
    count_moved = 0

    # 预先遍历以避免在 os.walk 中修改目录导致的问题
    all_files = list(get_image_files(args.path))
    
    for file_path in all_files:
        # 跳过目标垃圾桶目录
        if "_excluded_portrait" in file_path:
            continue
            
        count_scanned += 1
        try:
            with Image.open(file_path) as img:
                w, h = img.size
            
            if w <= h:
                # 保持相对目录结构
                rel_path = os.path.relpath(os.path.dirname(file_path), args.path)
                target_dir = os.path.join(trash_dir, rel_path)
                os.makedirs(target_dir, exist_ok=True)
                
                shutil.move(file_path, os.path.join(target_dir, os.path.basename(file_path)))
                count_moved += 1
                sys.stdout.write(f"\rMoved: {count_moved} / Scanned: {count_scanned}")
                sys.stdout.flush()

        except Exception:
            continue

    print(f"\nDone. Scanned: {count_scanned}, Moved: {count_moved}")

def cmd_dedupe(args):
    """去重模式 (基于 dHash)"""
    trash_dir = os.path.join(args.path, "_duplicates_removed")
    if not args.dry_run:
        os.makedirs(trash_dir, exist_ok=True)

    print(f"Mode: Deduplication | Threshold: {args.threshold} | Dry-run: {args.dry_run}")
    print(f"Path: {args.path}")

    # 获取所有文件并排序，确保处理顺序一致
    all_files = sorted(list(get_image_files(args.path)))
    
    seen_hashes = [] # 格式: (hash_obj, file_path)
    count_removed = 0
    count_kept = 0
    total = len(all_files)

    for i, file_path in enumerate(all_files):
        # 跳过垃圾桶目录
        if "_duplicates_removed" in file_path:
            continue

        try:
            with Image.open(file_path) as img:
                curr_hash = imagehash.dhash(img)
            
            is_duplicate = False
            
            # 与已有哈希库对比
            # 注意：对于超大数据集，这里可以使用 VP-Tree 或 BK-Tree 优化，
            # 但对于一般数据集，线性扫描列表通常足够快。
            for old_hash, old_path in seen_hashes:
                if (curr_hash - old_hash) <= args.threshold:
                    is_duplicate = True
                    break
            
            if is_duplicate:
                count_removed += 1
                if not args.dry_run:
                    shutil.move(file_path, os.path.join(trash_dir, os.path.basename(file_path)))
            else:
                seen_hashes.append((curr_hash, file_path))
                count_kept += 1
            
            sys.stdout.write(f"\rScanning: {i+1}/{total} | Kept: {count_kept} | Duplicates: {count_removed}")
            sys.stdout.flush()

        except Exception as e:
            continue

    print(f"\nDone. Kept: {count_kept}, Removed: {count_removed}")

def main():
    parser = argparse.ArgumentParser(description="Image Dataset Processing Utility")
    subparsers = parser.add_subparsers(dest='command', required=True, help="Available commands")

    # 1. Trim 命令配置
    p_trim = subparsers.add_parser('trim', help="Trim black borders from images")
    p_trim.add_argument('path', help="Dataset root directory")
    p_trim.add_argument('--threshold', type=int, default=10, help="Pixel brightness threshold (0-255, default: 10)")
    p_trim.set_defaults(func=cmd_trim)

    # 2. Filter 命令配置
    p_filter = subparsers.add_parser('filter', help="Move non-landscape images (Width <= Height)")
    p_filter.add_argument('path', help="Dataset root directory")
    p_filter.set_defaults(func=cmd_filter)

    # 3. Dedupe 命令配置
    p_dedupe = subparsers.add_parser('dedupe', help="Remove duplicate images using dHash")
    p_dedupe.add_argument('path', help="Dataset root directory")
    p_dedupe.add_argument('--threshold', type=int, default=3, help="Hash distance threshold (default: 3)")
    p_dedupe.add_argument('--dry-run', action='store_true', help="Run without moving files")
    p_dedupe.set_defaults(func=cmd_dedupe)

    args = parser.parse_args()
    
    if not os.path.exists(args.path):
        print(f"Error: Path '{args.path}' not found.")
        sys.exit(1)

    # 执行对应的函数
    args.func(args)

if __name__ == "__main__":
    main()