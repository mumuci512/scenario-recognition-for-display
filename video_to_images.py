import cv2
import os
import argparse
import imagehash
from PIL import Image

def extract_unique_frames(video_path, output_folder, class_name, interval=10, threshold=6):
    if not os.path.exists(video_path):
        print(f"Error: File not found: {video_path}")
        return

    save_dir = os.path.join(output_folder, class_name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    
    print(f"Processing: {video_basename} | Class: {class_name}")
    print(f"Config: Interval={interval}, Threshold={threshold}")

    frame_idx = 0
    saved_count = 0
    seen_hashes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 间隔跳帧，减少计算量
        if frame_idx % interval == 0:
            # OpenCV (BGR) -> PIL (RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # 计算平均哈希
            curr_hash = imagehash.average_hash(pil_img)
            
            is_unique = True
            # 与所有已保存的帧进行指纹比对
            for old_hash in seen_hashes:
                if abs(curr_hash - old_hash) < threshold:
                    is_unique = False
                    break 
            
            if is_unique:
                filename = f"{class_name}_{video_basename}_{saved_count:05d}.jpg"
                save_path = os.path.join(save_dir, filename)
                cv2.imwrite(save_path, frame)
                
                seen_hashes.append(curr_hash)
                saved_count += 1
                
                if saved_count % 10 == 0:
                    print(f"  -> Saved {saved_count} unique frames...")

        frame_idx += 1

    cap.release()
    print(f"Done. Processed {frame_idx} frames, saved {saved_count} unique images.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video frame extraction with duplicate removal.")
    
    parser.add_argument('video_path', type=str, help="Path to the input video file")
    parser.add_argument('--class_name', type=str, required=True, help="Category name (e.g., 'Word', 'CSGO')")
    parser.add_argument('--output', type=str, default='./dataset', help="Root output directory")
    parser.add_argument('--interval', type=int, default=15, help="Frame check interval (default: 15)")
    parser.add_argument('--threshold', type=int, default=6, help="Hash similarity threshold (default: 6)")

    args = parser.parse_args()

    extract_unique_frames(
        video_path=args.video_path,
        output_folder=args.output,
        class_name=args.class_name,
        interval=args.interval,
        threshold=args.threshold
    )