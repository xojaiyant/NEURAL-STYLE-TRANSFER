import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(img_path, max_size=400, shape=None):
    image = Image.open(img_path).convert('RGB')

    size = max_size if max(image.size) > max_size else max(image.size)
    if shape:
        size = shape

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = transform(image)[:3, :, :].unsqueeze(0)
    return image.to(device)


def im_convert(tensor):
    image = tensor.to("cpu").clone().detach().squeeze(0)
    image = image.numpy().transpose(1, 2, 0)
    image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    return image.clip(0, 1)


content = load_image("content.jpg")
style = load_image("style.jpg", shape=content.shape[-2:])
s
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Content Image")
plt.imshow(im_convert(content))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title("Style Image")
plt.imshow(im_convert(style))
plt.axis('off')
plt.show()


vgg = models.vgg19(pretrained=True).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False

def get_features(image, model):
    layers = {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2',  
        '28': 'conv5_1'
    }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    return torch.mm(tensor, tensor.t())

content_features = get_features(content, vgg)
style_features = get_features(style, vgg)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

target = content.clone().requires_grad_(True).to(device)


style_weights = {
    'conv1_1': 1.0,
    'conv2_1': 0.8,
    'conv3_1': 0.5,
    'conv4_1': 0.3,
    'conv5_1': 0.1
}

content_weight = 1e4
style_weight = 1e2

optimizer = optim.Adam([target], lr=0.003)

print("Starting Style Transfer...")
for step in range(1, 1501):
    target_features = get_features(target, vgg)
    
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
    
    style_loss = 0
    for layer in style_weights:
        target_feat = target_features[layer]
        target_gram = gram_matrix(target_feat)
        style_gram = style_grams[layer]
        _, d, h, w = target_feat.shape
        style_loss += style_weights[layer] * torch.mean((target_gram - style_gram)**2) / (d * h * w)
    
    total_loss = content_weight * content_loss + style_weight * style_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 300 == 0:
        print(f"Step {step}, Total Loss: {total_loss.item():.4f}")
        plt.imshow(im_convert(target))
        plt.title(f"Step {step}")
        plt.axis('off')
        plt.show()

output = im_convert(target)
plt.imsave("stylized_output.jpg", output)
print("Stylized image saved as stylized_output.jpg")
