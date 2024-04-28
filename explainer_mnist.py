import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import shap
import os

from dataset_v2 import ASL_MNIST
from models import CNN_MNIST

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(path):
    model = CNN_MNIST().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def main():
    # Initialize the model
    model = load_model(os.path.join('checkpoints', 'asl_mnist_cnn.pth'))

    train_dataset = ASL_MNIST(csv_file=os.path.join('ASL_MNIST', 'train', 'sign_mnist.csv'))
    test_dataset = ASL_MNIST(csv_file=os.path.join('ASL_MNIST', 'test', 'sign_mnist.csv'))
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    images_background, _ = next(iter(train_loader))
    images_background.to(device)
    #print(f"IMg: {images_background.shape}")
    
    # Wrap the model with a SHAP GradientExplainer
    explainer = shap.GradientExplainer(model, images_background)

    # Select a single batch for explanation
    images, labels = next(iter(test_loader))
    images = images.to(device)

    pred = model(images)
    predicted_label = pred.argmax(dim=1, keepdim=True).numpy()

    # Compute SHAP values
    shap_values = explainer.shap_values(images, nsamples=20)

    image_to_plot = -images[0].permute(1,2,0).cpu().numpy()
    label_to_plot = labels.numpy()[0]
    class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

    for i in range(shap_values.shape[-1]):
        shap_values_to_plot = shap_values[0,:,:,:,i].transpose(1,2,0)
        shap.image_plot(shap_values_to_plot, image_to_plot, show=False)
        plt.title(f"Class {i} - {class_names[i]}")
        plt.suptitle(f"True Label: {label_to_plot} --- Predicted Label: {predicted_label}")
        plt.savefig(os.path.join('plots',f'class_{i}.png'))
        #plt.show()
        plt.close()


if __name__ == '__main__':
    main()
