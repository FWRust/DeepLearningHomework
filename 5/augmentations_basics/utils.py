import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import psutil
from torchvision import transforms

def show_images(images, labels=None, nrow=8, title=None, size=128):
    """Визуализирует батч изображений."""
    images = images[:nrow]
    
    # Увеличиваем изображения до 128x128 для лучшей видимости
    resize_transform = transforms.Resize((size, size), antialias=True)
    images_resized = [resize_transform(img) for img in images]
    
    # Создаем сетку изображений
    fig, axes = plt.subplots(1, nrow, figsize=(nrow*2, 2))
    if nrow == 1:
        axes = [axes]
    
    for i, img in enumerate(images_resized):
        img_np = img.numpy().transpose(1, 2, 0)
        # Нормализуем для отображения
        img_np = np.clip(img_np, 0, 1)
        axes[i].imshow(img_np)
        axes[i].axis('off')
        if labels is not None:
            axes[i].set_title(f'Label: {labels[i]}')
    
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def show_single_augmentation(original_img, augmented_img, title="Аугментация"):
    """Визуализирует оригинальное и аугментированное изображение рядом."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    # Увеличиваем изображения
    resize_transform = transforms.Resize((128, 128), antialias=True)
    orig_resized = resize_transform(original_img)
    aug_resized = resize_transform(augmented_img)
    
    # Оригинальное изображение
    orig_np = orig_resized.numpy().transpose(1, 2, 0)
    orig_np = np.clip(orig_np, 0, 1)
    ax1.imshow(orig_np)
    ax1.set_title("Оригинал")
    ax1.axis('off')
    
    # Аугментированное изображение
    aug_np = aug_resized.numpy().transpose(1, 2, 0)
    aug_np = np.clip(aug_np, 0, 1)
    ax2.imshow(aug_np)
    ax2.set_title(title)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

def show_multiple_augmentations(original_img, augmented_imgs, titles):
    """Визуализирует оригинальное изображение и несколько аугментаций."""
    n_augs = len(augmented_imgs)
    fig, axes = plt.subplots(1, n_augs + 1, figsize=((n_augs + 1) * 2, 2))
    
    # Увеличиваем изображения
    resize_transform = transforms.Resize((128, 128), antialias=True)
    orig_resized = resize_transform(original_img)
    
    # Оригинальное изображение
    orig_np = orig_resized.numpy().transpose(1, 2, 0)
    orig_np = np.clip(orig_np, 0, 1)
    axes[0].imshow(orig_np)
    axes[0].set_title("Оригинал")
    axes[0].axis('off')
    
    # Аугментированные изображения
    for i, (aug_img, title) in enumerate(zip(augmented_imgs, titles)):
        aug_resized = resize_transform(aug_img)
        aug_np = aug_resized.numpy().transpose(1, 2, 0)
        aug_np = np.clip(aug_np, 0, 1)
        axes[i + 1].imshow(aug_np)
        axes[i + 1].set_title(title)
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.show() 

def count_images_in_classes(dataset):
    count = 0
    curr_class = 0
    class_count = {}
    for image, label in dataset:
        if label == curr_class:
            count += 1
        else:
            class_count[curr_class] = count
            curr_class += 1
            count = 1
    class_count[curr_class] = count
    return class_count

def analyze_image_sizes(dataset):
    """Анализирует размеры изображений в датасете"""
    sizes = [(img.size[0], img.size[1]) for img, _ in dataset]
    widths, heights = zip(*sizes)
    return {
        'min': (min(widths), min(heights)),
        'max': (max(widths), max(heights)),
        'avg': (sum(widths)//len(widths), sum(heights)//len(heights))
    }
    
def visualize_dataset_stats(dataset):
    """Визуализирует распределение размеров и гистограмму по классам."""
    # Подсчет по классам
    class_counts = count_images_in_classes(dataset)
    
    # Анализ размеров
    sizes = [(img.size[0], img.size[1]) for img, _ in dataset]
    widths, heights = zip(*sizes)
    
    # Создаем график
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Гистограмма по классам
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    ax1.bar(classes, counts)
    ax1.set_title('Распределение по классам')
    ax1.set_xlabel('Класс')
    ax1.set_ylabel('Количество изображений')
    
    # Распределение размеров
    ax2.scatter(widths, heights, alpha=0.6)
    ax2.set_title('Распределение размеров изображений')
    ax2.set_xlabel('Ширина')
    ax2.set_ylabel('Высота')
    
    plt.tight_layout()
    plt.show()

class AugmentationPipeline:
    """Класс для управления пайплайном аугментаций."""
    
    def __init__(self):
        self.augmentations = {}
        self.pipeline = None
    
    def add_augmentation(self, name, aug):
        """Добавляет аугментацию в пайплайн."""
        self.augmentations[name] = aug
        self._update_pipeline()
    
    def remove_augmentation(self, name):
        """Удаляет аугментацию из пайплайна."""
        if name in self.augmentations:
            del self.augmentations[name]
            self._update_pipeline()
        else:
            print(f"Аугментация {name} не найдена в пайплайне.")
    
    def apply(self, image):
        """Применяет пайплайн аугментаций к изображению."""
        if self.pipeline is None:
            return image
        return self.pipeline(image)
    
    def get_augmentations(self):
        """Возвращает словарь всех аугментаций в пайплайне."""
        return self.augmentations
    
    def _update_pipeline(self):
        """Обновляет внутренний пайплайн на основе текущих аугментаций."""
        if not self.augmentations:
            self.pipeline = None
        else:
            aug_list = list(self.augmentations.values())
            self.pipeline = transforms.Compose(aug_list) if len(aug_list) > 1 else aug_list[0]


def measure_performance(size, num_images=100):
    """Измеряет время и память для заданного размера изображений"""
    
    # Создаем аугментации
    augs = [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
    ]
    
    transform = transforms.Compose(augs)
    
    # Загружаем датасет
    from augmentations_basics.datasets import CustomImageDataset
    dataset = CustomImageDataset('data/train', transform=None, target_size=size)
    
    # Измеряем время загрузки
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    images = []
    for i in range(min(num_images, len(dataset))):
        img, _ = dataset[i]
        images.append(img)
    
    load_time = time.time() - start_time
    load_memory = psutil.Process().memory_info().rss / 1024 / 1024 - start_memory
    
    # Измеряем время аугментаций
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    augmented_images = []
    for img in images:
        aug_img = transform(img)
        augmented_images.append(aug_img)
    
    aug_time = time.time() - start_time
    aug_memory = psutil.Process().memory_info().rss / 1024 / 1024 - start_memory
    
    return {
        'load_time': load_time,
        'aug_time': aug_time,
        'total_time': load_time + aug_time,
        'load_memory': load_memory,
        'aug_memory': aug_memory,
        'total_memory': load_memory + aug_memory
    }


def run_size_experiment():
    """Запускает эксперимент с разными размерами"""
    
    sizes = [(64, 64), (128, 128), (224, 224), (512, 512)]
    results = {}
    
    print("Запуск эксперимента с размерами изображений...")
    print("=" * 50)
    
    for size in sizes:
        print(f"Тестируем размер: {size[0]}x{size[1]}")
        results[size] = measure_performance(size)
        
        print(f"  Время загрузки: {results[size]['load_time']:.3f} сек")
        print(f"  Время аугментаций: {results[size]['aug_time']:.3f} сек")
        print(f"  Общее время: {results[size]['total_time']:.3f} сек")
        print(f"  Память загрузки: {results[size]['load_memory']:.1f} MB")
        print(f"  Память аугментаций: {results[size]['aug_memory']:.1f} MB")
        print(f"  Общая память: {results[size]['total_memory']:.1f} MB")
        print("-" * 30)
    
    return results


def plot_size_experiment_results(results):
    """Строит графики результатов эксперимента с размерами"""
    
    sizes = list(results.keys())
    size_labels = [f"{s[0]}x{s[1]}" for s in sizes]
    
    # Время
    load_times = [results[s]['load_time'] for s in sizes]
    aug_times = [results[s]['aug_time'] for s in sizes]
    total_times = [results[s]['total_time'] for s in sizes]
    
    # Память
    load_memory = [results[s]['load_memory'] for s in sizes]
    aug_memory = [results[s]['aug_memory'] for s in sizes]
    total_memory = [results[s]['total_memory'] for s in sizes]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # График времени
    x = np.arange(len(size_labels))
    width = 0.25
    
    ax1.bar(x - width, load_times, width, label='Загрузка', alpha=0.8)
    ax1.bar(x, aug_times, width, label='Аугментации', alpha=0.8)
    ax1.bar(x + width, total_times, width, label='Общее время', alpha=0.8)
    
    ax1.set_xlabel('Размер изображения')
    ax1.set_ylabel('Время (секунды)')
    ax1.set_title('Время обработки по размерам')
    ax1.set_xticks(x)
    ax1.set_xticklabels(size_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График памяти
    ax2.bar(x - width, load_memory, width, label='Загрузка', alpha=0.8)
    ax2.bar(x, aug_memory, width, label='Аугментации', alpha=0.8)
    ax2.bar(x + width, total_memory, width, label='Общая память', alpha=0.8)
    
    ax2.set_xlabel('Размер изображения')
    ax2.set_ylabel('Память (MB)')
    ax2.set_title('Потребление памяти по размерам')
    ax2.set_xticks(x)
    ax2.set_xticklabels(size_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # График зависимости времени от размера
    pixel_counts = [s[0] * s[1] for s in sizes]
    ax3.scatter(pixel_counts, total_times, s=100, alpha=0.7)
    ax3.plot(pixel_counts, total_times, 'r--', alpha=0.7)
    ax3.set_xlabel('Количество пикселей')
    ax3.set_ylabel('Время (секунды)')
    ax3.set_title('Зависимость времени от размера')
    ax3.grid(True, alpha=0.3)
    
    # График зависимости памяти от размера
    ax4.scatter(pixel_counts, total_memory, s=100, alpha=0.7)
    ax4.plot(pixel_counts, total_memory, 'r--', alpha=0.7)
    ax4.set_xlabel('Количество пикселей')
    ax4.set_ylabel('Память (MB)')
    ax4.set_title('Зависимость памяти от размера')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('size_experiment_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Выводим таблицу результатов
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА")
    print("=" * 80)
    print(f"{'Размер':<12} {'Время загрузки':<15} {'Время аугментаций':<18} {'Общее время':<12} {'Память (MB)':<12}")
    print("-" * 80)
    
    for size in sizes:
        size_str = f"{size[0]}x{size[1]}"
        print(f"{size_str:<12} {results[size]['load_time']:<15.3f} {results[size]['aug_time']:<18.3f} "
              f"{results[size]['total_time']:<12.3f} {results[size]['total_memory']:<12.1f}") 


