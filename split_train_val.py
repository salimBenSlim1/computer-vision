import os
import shutil

def split_train_val(img_src='coffee_dataset/images/train', img_dst='coffee_dataset/images/val',
                    lbl_src='coffee_dataset/labels/train', lbl_dst='coffee_dataset/labels/val',
                    step=6):
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(lbl_dst, exist_ok=True)

    imgs = sorted([f for f in os.listdir(img_src)
                   if os.path.isfile(os.path.join(img_src, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    moved = 0
    for i, fname in enumerate(imgs):
        if i % step == 0:
            src_path = os.path.join(img_src, fname)
            dst_path = os.path.join(img_dst, fname)
            shutil.move(src_path, dst_path)

            lbl_name = os.path.splitext(fname)[0] + '.txt'
            src_lbl = os.path.join(lbl_src, lbl_name)
            dst_lbl = os.path.join(lbl_dst, lbl_name)
            if os.path.exists(src_lbl):
                shutil.move(src_lbl, dst_lbl)

            moved += 1

    print(f'Moved {moved} images (every {step}th) from {img_src} -> {img_dst}')


if __name__ == '__main__':
    split_train_val()
