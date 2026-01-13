from matplotlib import pyplot as plt

################### EXPLORATION DES DONNEES ###################

def show_images_by_style(dataset, target_style, n=5):
    count = 0
    for item in dataset:
        if item["style"] == target_style:
            plt.figure(figsize=(3,3))
            plt.imshow(item["image"])
            plt.axis("off")
            plt.title(f"{item['artist']} – {item['style']}")
            plt.show()
            count += 1
            if count == n:
                break