import os
import zipfile
import tarfile
import pickle
import streamlit as st
import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image

# --- Constants ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(normalize_option='imagenet', custom_mean=None, custom_std=None):
    """지정된 정규화 옵션에 따라 데이터 변환을 생성합니다."""
    
    mean, std = IMAGENET_MEAN, IMAGENET_STD  # 기본값
    
    if normalize_option == 'dataset':
        if custom_mean is None or custom_std is None:
            # 이 경우는 일반적으로 발생해서는 안 되지만, 안전장치로 추가
            st.warning("데이터셋 통계가 제공되지 않아 ImageNet 통계를 대신 사용합니다.")
        else:
            mean, std = custom_mean, custom_std
    
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def calculate_mean_std(dataset):
    """데이터셋의 평균과 표준편차를 계산합니다."""
    # 정규화 없이 텐서로만 변환
    temp_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # 원본 데이터셋의 transform을 임시로 변경
    original_transform = dataset.transform
    dataset.transform = temp_transform

    # 계산을 위한 데이터로더
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    
    mean = 0.
    std = 0.
    total_images_count = 0
    
    pbar = st.progress(0, text="데이터셋 평균/표준편차 계산 중...")
    for i, (images, _) in enumerate(loader):
        batch_samples = images.size(0) # 배치 크기
        images = images.view(batch_samples, images.size(1), -1) # (N, C, H*W)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples
        
        progress = (i + 1) / len(loader)
        pbar.progress(progress, text=f"데이터셋 통계 계산 중... {int(progress*100)}%")
        
    pbar.empty()

    mean /= total_images_count
    std /= total_images_count
    
    # 데이터셋의 transform을 원상복구
    dataset.transform = original_transform
    
    st.success(f"데이터셋 통계 계산 완료: Mean={mean.tolist()}, Std={std.tolist()}")
    
    return mean.tolist(), std.tolist()


# --- Custom Dataset for Pickle ---
class PickledDataset(Dataset):
    """Pickle 파일에서 로드된 NumPy 배열을 위한 커스텀 데이터셋"""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        # PIL Image로 변환하여 torchvision transform과 호환되도록 함
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# --- Main Entry Point ---
def create_dataloaders(uploaded_file, upload_dir, batch_size, subset_ratio, random_state, normalize_option):
    """Uploaded file을 분석하여 형식에 맞는 데이터로더를 생성하는 메인 함수"""
    if uploaded_file is None: return None, None, None

    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    # Handle .tar.gz
    if uploaded_file.name.endswith('.tar.gz'): file_ext = '.tar.gz'

    if file_ext in ['.zip', '.tar.gz']:
        extract_path = handle_archive_upload(uploaded_file, upload_dir)
        if not extract_path: return None, None, None
        
        data_root = find_data_root(extract_path)
        dataset_format = detect_dataset_format(data_root)
        st.info(f"감지된 데이터셋 형식: **{dataset_format}**")

        if dataset_format == "ImageFolder":
            return get_loaders_from_image_folder(data_root, batch_size, subset_ratio, random_state, normalize_option)
        elif dataset_format == "CIFAR-10":
            return get_loaders_from_cifar_style(data_root, batch_size, subset_ratio, random_state, normalize_option)
        else:
            st.error("압축 파일 내에서 지원하는 데이터셋 구조(ImageFolder, CIFAR-10)를 찾지 못했습니다.")
            return None, None, None

    elif file_ext in ['.pkl', '.pickle']:
        st.info(f"감지된 데이터셋 형식: **Monolithic Pickle**")
        return get_loaders_from_pickle(uploaded_file, upload_dir, batch_size, subset_ratio, random_state, normalize_option)
    else:
        st.error(f"지원하지 않는 파일 형식입니다: {file_ext}")
        return None, None, None

# --- Format-Specific Loader Functions ---

def get_loaders_from_image_folder(data_dir, batch_size, subset_ratio, random_state, normalize_option):
    """ImageFolder 구조로부터 데이터로더 생성"""
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        st.error(f"'{train_dir}'와 '{val_dir}' 폴더를 찾을 수 없습니다.")
        return None, None, None

    # 1. Transform 없이 데이터셋 우선 로드
    full_train_dataset = datasets.ImageFolder(train_dir, transform=None)
    
    # 2. 정규화 옵션에 따라 Transform 결정
    custom_mean, custom_std = None, None
    if normalize_option == 'dataset':
        # 임시 데이터셋(서브셋)을 만들어 통계 계산 (전체는 너무 오래 걸릴 수 있음)
        temp_train_ds, _ = _apply_subset(full_train_dataset, datasets.ImageFolder(val_dir, transform=None), 0.2, random_state)
        st.info(f"정확한 통계 계산을 위해 학습 데이터의 20%({len(temp_train_ds)}개)를 사용합니다.")
        custom_mean, custom_std = calculate_mean_std(temp_train_ds)

    train_transform = get_transforms(normalize_option, custom_mean, custom_std)
    val_transform = get_transforms(normalize_option, custom_mean, custom_std) # 검증셋에도 동일 적용
    
    # 3. 최종 Transform을 데이터셋에 적용
    full_train_dataset.transform = train_transform
    full_val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    # 4. 데이터셋 서브셋 적용 및 로더 생성
    train_dataset, val_dataset = _apply_subset(
        full_train_dataset, full_val_dataset, subset_ratio, random_state
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, len(full_train_dataset.classes)

def get_loaders_from_cifar_style(data_dir, batch_size, subset_ratio, random_state, normalize_option):
    """CIFAR-10 스타일 구조로부터 데이터로더 생성"""
    cifar_root = data_dir
    if os.path.basename(data_dir) in ['cifar-10-batches-py', 'cifar-100-python']:
        cifar_root = os.path.dirname(data_dir)

    try:
        # 1. Transform 없이 데이터셋 우선 로드
        full_train_dataset = datasets.CIFAR10(root=cifar_root, train=True, download=False, transform=None)
        full_val_dataset = datasets.CIFAR10(root=cifar_root, train=False, download=False, transform=None)
    except RuntimeError as e:
        st.error(f"CIFAR-10 데이터셋 로드에 실패했습니다. 파일이 손상되었거나 구조가 올바르지 않을 수 있습니다.")
        st.info(f"에러: {e}")
        st.info(f"기대한 데이터셋 루트 경로: {cifar_root}")
        return None, None, None

    # 2. 정규화 옵션에 따라 Transform 결정
    custom_mean, custom_std = None, None
    if normalize_option == 'dataset':
        # CIFAR-10은 크기가 표준화되어 있으므로 전체 학습셋으로 계산
        st.info(f"정확한 통계 계산을 위해 전체 학습 데이터({len(full_train_dataset)}개)를 사용합니다.")
        custom_mean, custom_std = calculate_mean_std(full_train_dataset)

    final_transform = get_transforms(normalize_option, custom_mean, custom_std)

    # 3. 최종 Transform을 데이터셋에 적용
    full_train_dataset.transform = final_transform
    full_val_dataset.transform = final_transform

    # 4. 데이터셋 서브셋 적용 및 로더 생성
    train_dataset, val_dataset = _apply_subset(
        full_train_dataset, full_val_dataset, subset_ratio, random_state
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, len(full_train_dataset.classes)

def get_loaders_from_pickle(uploaded_file, upload_dir, batch_size, subset_ratio, random_state, normalize_option):
    """단일 Pickle 파일로부터 데이터로더 생성"""
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        st.error(f"Pickle 파일을 읽는 중 오류가 발생했습니다: {e}")
        return None, None, None

    required_keys = ['train_images', 'train_labels', 'val_images', 'val_labels', 'classes']
    if not all(key in data for key in required_keys):
        st.error(f"Pickle 파일에 필요한 키가 모두 포함되어 있지 않습니다. ({required_keys})")
        return None, None, None

    # 1. Transform 없이 데이터셋 우선 로드
    full_train_dataset = PickledDataset(data['train_images'], data['train_labels'], transform=None)
    full_val_dataset = PickledDataset(data['val_images'], data['val_labels'], transform=None)

    # 2. 정규화 옵션에 따라 Transform 결정
    custom_mean, custom_std = None, None
    if normalize_option == 'dataset':
        st.info(f"정확한 통계 계산을 위해 전체 학습 데이터({len(full_train_dataset)}개)를 사용합니다.")
        custom_mean, custom_std = calculate_mean_std(full_train_dataset)
    
    final_transform = get_transforms(normalize_option, custom_mean, custom_std)
    
    # 3. 최종 Transform을 데이터셋에 적용
    full_train_dataset.transform = final_transform
    full_val_dataset.transform = final_transform

    # 4. 데이터셋 서브셋 적용 및 로더 생성
    train_dataset, val_dataset = _apply_subset(
        full_train_dataset, full_val_dataset, subset_ratio, random_state
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, len(data['classes'])

# --- Helper Functions ---

def _apply_subset(train_ds, val_ds, ratio, seed):
    """데이터셋에 서브셋 샘플링을 적용하는 내부 함수"""
    if ratio < 1.0:
        st.info(f"전체 데이터의 {ratio * 100:.0f}%만 사용하여 학습 및 평가를 진행합니다.")
        
        # Train
        train_targets = train_ds.targets if hasattr(train_ds, 'targets') else [s[1] for s in train_ds.samples]
        train_indices, _ = train_test_split(range(len(train_ds)), train_size=ratio, stratify=train_targets, random_state=seed)
        train_ds = Subset(train_ds, train_indices)
        
        # Validation
        val_targets = val_ds.targets if hasattr(val_ds, 'targets') else [s[1] for s in val_ds.samples]
        if len(val_targets) > 0:
            val_indices, _ = train_test_split(range(len(val_ds)), train_size=ratio, stratify=val_targets, random_state=seed)
            val_ds = Subset(val_ds, val_indices)
    return train_ds, val_ds

def detect_dataset_format(data_dir):
    """디렉토리 구조를 보고 데이터셋 포맷을 추측"""
    dir_content = os.listdir(data_dir)
    if 'batches.meta' in dir_content and 'data_batch_1' in dir_content:
        return "CIFAR-10"
    if 'train' in dir_content and 'val' in dir_content:
        if os.path.isdir(os.path.join(data_dir, 'train')) and os.path.isdir(os.path.join(data_dir, 'val')):
            return "ImageFolder"
    return "Unknown"

def find_data_root(extract_path):
    """압축 해제된 폴더 내에서 실제 데이터가 담긴 루트 폴더를 탐색"""
    for root, dirs, files in os.walk(extract_path):
        if ('train' in dirs and 'val' in dirs) or 'batches.meta' in files:
            return root
    return extract_path

def handle_archive_upload(uploaded_file, upload_dir='uploads'):
    """압축 파일을 저장하고 지정된 경로에 압축 해제"""
    if not os.path.exists(upload_dir): os.makedirs(upload_dir)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    extract_base_name = os.path.splitext(os.path.splitext(uploaded_file.name)[0])[0]
    extract_path = os.path.join(upload_dir, extract_base_name)
    if not os.path.exists(extract_path): os.makedirs(extract_path)

    try:
        if uploaded_file.name.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        elif uploaded_file.name.endswith('.tar.gz'):
            with tarfile.open(file_path, 'r:gz') as tar_ref:
                tar_ref.extractall(path=extract_path)
        st.success(f"'{uploaded_file.name}' 파일이 업로드 및 압축 해제되었습니다.")
    except Exception as e:
        st.error(f"압축 해제 중 오류가 발생했습니다: {e}")
        return None
    
    return extract_path
