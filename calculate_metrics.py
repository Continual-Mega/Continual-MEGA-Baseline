import csv
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Calculate metrics from a CSV file.')
    parser.add_argument("--image_csv", type=str, help="Path to the image CSV file.", required=True)
    parser.add_argument("--pixel_csv", type=str, help="Path to the pixel CSV file.", required=True)
    
    args = parser.parse_args()

    results_image = []
    results_pixel = []
    with open(args.image_csv, mode="r") as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            results_image.append([float(x) for x in row])
    with open(args.pixel_csv, mode="r") as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            results_pixel.append([float(x) for x in row])

    results_image = np.array(results_image)
    results_pixel = np.array(results_pixel)
    image_max = np.max(results_image[:-1, :-1], axis=0)
    pixel_max = np.max(results_pixel[:-1, :-1], axis=0)

    image_acc = round(np.mean(results_image[-1]) * 100, 1)
    image_fm = round(np.mean(image_max - results_image[-1:,:-1]) * 100, 1)
    pixel_acc = round(np.mean(results_pixel[-1]) * 100, 1)
    pixel_fm = round(np.mean(pixel_max - results_pixel[-1:,:-1]) * 100, 1)
    
    print(f"Image ACC: {image_acc}")
    print(f"Image FM: {image_fm}")
    print(f"Pixel ACC: {pixel_acc}")
    print(f"Pixel FM: {pixel_fm}")

if __name__ == "__main__":
    main()