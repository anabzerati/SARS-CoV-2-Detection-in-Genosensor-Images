from torchvision import transforms

# Standard normalization (ImageNet)
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Augmentation pipelines
augmentations = {
    "None": transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        NORMALIZE
    ]),

    "Flipping": transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        NORMALIZE
    ]),

    "Shift": transforms.Compose([
        transforms.Resize(224),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        NORMALIZE
    ]),

    "RandomErasing": transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        NORMALIZE,
        transforms.RandomErasing(p=0.5)
    ]),

    "AutoAugment": transforms.Compose([
        transforms.Resize(224),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        NORMALIZE
    ]),

    "RandAugment": transforms.Compose([
        transforms.Resize(224),
        transforms.RandAugment(),
        transforms.ToTensor(),
        NORMALIZE
    ]),

    "TrivialAugment": transforms.Compose([
        transforms.Resize(224),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        NORMALIZE
    ]),

    "AugMix": transforms.Compose([
        transforms.Resize(224),
        transforms.AugMix(),
        transforms.ToTensor(),
        NORMALIZE
    ])
}

def get_augmentation(name: str):
    """
    Retrieve a torchvision.transforms augmentation by name.
    
    Args:
        name (str): 
            Name of the augmentation, must be a key in augmentations.
        
    Returns:
        torchvision.transforms.Compose
    """
    if name not in augmentations:
        raise ValueError(f"Augmentation '{name}' not found. Available: {list(augmentations.keys())}")
    return augmentations[name]
