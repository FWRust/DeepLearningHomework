from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import os
from augmentations_basics.datasets import CustomImageDataset
from augmentations_basics.utils import *
from augmentations_basics.extra_augs import *
from torch.utils.data import DataLoader
### 1
augs = [
    ("RandomHorizontalFlip", transforms.RandomHorizontalFlip(p=1.0)),
    ("RandomCrop", transforms.RandomCrop(200, padding=20)),
    ("ColorJitter", transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)),
    ("RandomRotation", transforms.RandomRotation(degrees=30)),
    ("RandomGrayscale", transforms.RandomGrayscale(p=1.0))
]
augmentation_pipeline = transforms.Compose(augs)




root = 'data/train'
dataset = CustomImageDataset(root, transform=None, target_size=(224, 224))
class_names = dataset.get_class_names()

examples = []
label_find = 0
for i in dataset:
    original_img, label = i
    if label_find == label:
        examples.append(original_img)
        label_find += 1
        if label_find == len(class_names):
            break

augmented_imgs = []
titles = []
for img in examples:
    augmented_imgs = []
    titles = []
    for name, aug in augs:
        aug_transform = transforms.Compose([
            aug,
            transforms.ToTensor()
        ])
        
        aug_img = aug_transform(img)
        augmented_imgs.append(aug_img)
        titles.append(name)
    aug_full_transform = transforms.Compose([aug[1] for aug in augs] + [transforms.ToTensor()])
    aug_full_img = aug_full_transform(img)
    img = transforms.ToTensor()(img)
    show_multiple_augmentations(img, augmented_imgs, titles)
    show_single_augmentation(img, aug_full_img, "Full Augmentation")

### 2
augs = [
    transforms.ToTensor(),
    RandomBrightness(p=1.0),
    RandomPixelate(p=1.0)
]
augmentation_pipeline = transforms.Compose(augs)

for example in examples:
    original_img = transforms.ToTensor()(example)
    aug_img = augmentation_pipeline(example)
    show_single_augmentation(original_img, aug_img, "Full Augmentation")

### 3
dataset = CustomImageDataset(root, transform=None)
# Подсчет количества изображений в каждом классе
class_count = count_images_in_classes(dataset)
print(class_count)

# Анализ размеров изображений
size_stats = analyze_image_sizes(dataset)
print(f"Минимальный размер: {size_stats['min']}")
print(f"Максимальный размер: {size_stats['max']}")
print(f"Средний размер: {size_stats['avg']}")

# Визуализация распределения размеров и гистограммы по классам
visualize_dataset_stats(dataset)

### 4
aug_pipeline = AugmentationPipeline()

light_config = [
    ("ToTensor", transforms.ToTensor()),
    ("RandomHorizontalFlip", transforms.RandomHorizontalFlip(p=0.5)),
    ("RandomRotation", transforms.RandomRotation(degrees=30)),
]
medium_config = [
    ("ToTensor", transforms.ToTensor()),
    ("RandomHorizontalFlip", transforms.RandomHorizontalFlip(p=0.65)),
    ("RandomRotation", transforms.RandomRotation(degrees=30)),
    ("ColorJitter", transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)),
    ("RandomBrightness", RandomBrightness(p=0.65)),
]
heavy_config = [
    ("ToTensor", transforms.ToTensor()),
    ("RandomHorizontalFlip", transforms.RandomHorizontalFlip(p=0.8)),
    ("RandomRotation", transforms.RandomRotation(degrees=30)),
    ("ColorJitter", transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)),
    ("RandomBrightness", RandomBrightness(p=0.8)),
    ("RandomPixelate", RandomPixelate(p=0.8)),
    ("RandomBlur", RandomBlur(p=0.8)),
    ("CutOut", CutOut(p=0.8))
]

light_aug_pipeline = AugmentationPipeline()
for aug in light_config:
    light_aug_pipeline.add_augmentation(aug[0], aug[1])

medium_aug_pipeline = AugmentationPipeline()
for aug in medium_config:
    medium_aug_pipeline.add_augmentation(aug[0], aug[1])

heavy_aug_pipeline = AugmentationPipeline()
for aug in heavy_config:
    heavy_aug_pipeline.add_augmentation(aug[0], aug[1])


dataset = CustomImageDataset(root, transform=None, target_size=(224, 224))
class_names = dataset.get_class_names()

# Сохранение аугментированных изображений
for idx in range(len(dataset)):
    image, label = dataset[idx]
    # Создаем папку для класса
    class_name = class_names[label]
    class_dir = os.path.join('5_augmented', class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    # Применяем аугментации
    light_aug_img = light_aug_pipeline.apply(image)
    medium_aug_img = medium_aug_pipeline.apply(image)
    heavy_aug_img = heavy_aug_pipeline.apply(image)
    
    # Преобразуем тензоры в PIL Image и сохраняем
    transforms.ToPILImage()(light_aug_img).save(os.path.join(class_dir, f'light_aug_{idx}.jpg'))
    transforms.ToPILImage()(medium_aug_img).save(os.path.join(class_dir, f'medium_aug_{idx}.jpg'))
    transforms.ToPILImage()(heavy_aug_img).save(os.path.join(class_dir, f'heavy_aug_{idx}.jpg'))

### 5 

results = run_size_experiment()


plot_size_experiment_results(results)

### 6 
# Подготовка датасета
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Создаем полный датасет
full_dataset = CustomImageDataset('5_augmented', transform=transform)

# Разделяем на тренировочный
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size]
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = torch.nn.Linear(model.fc.in_features, len(full_dataset.get_class_names()))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(10):
    # Обучение
    model.train()
    train_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            out = model(x)
            loss = loss_fn(out, y)
            val_loss += loss.item()
            
            # Вычисляем точность
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    train_losses.append(train_loss/len(train_loader))
    val_losses.append(val_loss/len(val_loader))
    val_accuracies.append(100*correct/total)
    
    print(f'Epoch {epoch+1}: Train Loss: {train_losses[-1]:.4f}, '
          f'Val Loss: {val_losses[-1]:.4f}, '
          f'Val Accuracy: {val_accuracies[-1]:.2f}%')

# Визуализация
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Val Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Validation Accuracy')

plt.tight_layout()
plt.show()