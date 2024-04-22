import torchvision

def crop_mouth(image):
    return torchvision.transforms.functional.crop(image, 230, 70, 250, 500)