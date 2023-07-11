import os
from PIL import Image, ImageFilter


def blurr_image():
    rgb_patch_path = 'datasets/SmallDataset2/RGB_Patch'
    blurred_patch_path = 'datasets/SmallDataset2/GBlurred_RGB_Patch'

    for i in range(3500):
        rgb = Image.open(f'{rgb_patch_path}/{i}.jpg')
        # grayscale_image = rgb.convert("L")
        blurred_image = rgb.filter(ImageFilter.GaussianBlur(radius=2))
        blurred_image.save(f"{blurred_patch_path}/{i}.jpg")


def divide_image():
    count = 0

    rgb_path = 'datasets/SmallDataset2/RGB'
    rgb_patch_path = 'datasets/SmallDataset2/RGB_Patch'
    hha_path = 'datasets/SmallDataset2/HHA'
    hha_patch_path = 'datasets/SmallDataset2/HHA_Patch'
    label_path = 'datasets/SmallDataset2/Label'
    label_patch_path = 'datasets/SmallDataset2/Label_Patch'
    colored_label_path = 'datasets/SmallDataset2/ColoredLabel'
    colored_label_patch_path = 'datasets/SmallDataset2/ColoredLabel_Patch'

    for i in range(40, 90, 1):
        rgb = Image.open(f'{rgb_path}/{i}.jpg')
        hha = Image.open(f'{hha_path}/{i}.jpg')
        label = Image.open(f'{label_path}/{i}.png')
        colored_label = Image.open(f'{colored_label_path}/{i}.png')

        width, height = rgb.size

        width = (width // 64) * 64
        height = (height // 64) * 64
    
        for y in range(0, height, 64):
            for x in range(0, width, 64):
                box = (x, y, x + 64, y + 64)
                r = rgb.crop(box)
                r.save(f"{rgb_patch_path}/{count}.jpg")
                h = hha.crop(box)
                h.save(f"{hha_patch_path}/{count}.jpg")
                l = label.crop(box)
                l.save(f"{label_patch_path}/{count}.png")
                cl = colored_label.crop(box)
                cl.save(f"{colored_label_patch_path}/{count}.png")
                print(count)
                count += 1

# divide_image()
blurr_image()