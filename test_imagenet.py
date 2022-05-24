from test_ood import test_ood

for m in "resnet18", "resnet34", "resnet50", "resnet101":
    test_ood("imagenet", m)
