import matplotlib.pyplot as plt

# convert from [C,H,W] to [H,W,C] for matplotlib
def show(tensor, title):
    img = tensor.permute(1, 2, 0).cpu().numpy()
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)