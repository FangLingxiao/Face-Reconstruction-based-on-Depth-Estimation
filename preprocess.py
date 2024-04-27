import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

class ImageProcessor:
    def __init__(self, model_path, input_folder, output_folder, device):
        self.model = torch.load(model_path) # Load pre-trained model
        self.model.to(device)
        self.model.eval()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.ToTensor(), # Convert image to PyTorch tensor
        ])

    def process_images(self):
        for filename in os.listdir(self.input_folder):
            if filename.endswith('.jpg'): # Check if the file is a JPEG image
                img_path = os.path.join(self.input_folder, filename)
                img = Image.open(img_path)
                img = self.transform(img).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output = self.model(img)

                vol = (output[0] * 255).byte().cpu()
                out_path = os.path.join(self.output_folder, filename[:-4] + '.raw')
                with open(out_path, 'wb') as f:
                    f.write(vol.numpy().tobytes())

                print(f'Processed {filename}')

if __name__ == "__main__":
    model_path = 'model.pth'
    input_folder = 'examples/'
    output_folder = 'output/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize and run the image processor
    processor = ImageProcessor(model_path, input_folder, output_folder, device)
    processor.process_images()