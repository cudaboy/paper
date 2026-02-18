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

# --- 상수 정의 ---
# ImageNet 데이터셋의 평균과 표준편차를 상수로 정의합니다.
# 이 값들은 이미지 정규화에 사용됩니다.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(normalize_option='imagenet', custom_mean=None, custom_std=None):
    """
    지정된 정규화 옵션에 따라 데이터 변환(transform) 파이프라인을 생성합니다.
    - `imagenet`: ImageNet의 통계 사용
    - `dataset`: 제공된 `custom_mean`, `custom_std` 사용
    """
    
    # 기본 정규화 값으로 ImageNet 통계를 설정합니다.
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    
    # 'dataset' 옵션이 선택된 경우, 사용자 정의 평균과 표준편차를 사용합니다.
    if normalize_option == 'dataset':
        if custom_mean is None or custom_std is None:
            # 사용자 정의 값이 제공되지 않은 경우 경고를 표시하고 ImageNet 통계를 사용합니다.
            st.warning("데이터셋 통계가 제공되지 않아 ImageNet 통계를 대신 사용합니다.")
        else:
            mean, std = custom_mean, custom_std
    
    # 이미지 크기 조정, 텐서 변환, 정규화를 포함하는 변환 파이프라인을 반환합니다.
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def calculate_mean_std(dataset):
    """
    주어진 데이터셋에 대해 픽셀 값의 채널별 평균과 표준편차를 계산합니다.
    계산 중에는 정규화 변환을 일시적으로 제거합니다.
    """
    # 평균 및 표준편차 계산을 위해 정규화가 없는 임시 변환을 정의합니다.
    temp_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # 데이터셋의 원래 변환을 저장하고 계산을 위해 임시 변환을 적용합니다.
    original_transform = dataset.transform
    dataset.transform = temp_transform

    # 전체 데이터셋을 순회하기 위한 데이터로더를 생성합니다.
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    
    mean = 0.
    std = 0.
    total_images_count = 0
    
    # 진행률 표시줄을 설정합니다.
    pbar = st.progress(0, text="데이터셋 평균/표준편차 계산 중...")
    # 데이터로더를 순회하며 모든 이미지에 대해 평균과 표준편차를 누적합니다.
    for i, (images, _) in enumerate(loader):
        batch_samples = images.size(0) # 현재 배치의 이미지 수
        images = images.view(batch_samples, images.size(1), -1) # (N, C, H*W) 형태로 변환
        mean += images.mean(2).sum(0) # 채널별 평균을 계산하고 누적
        std += images.std(2).sum(0)   # 채널별 표준편차를 계산하고 누적
        total_images_count += batch_samples
        
        # 진행률을 업데이트합니다.
        progress = (i + 1) / len(loader)
        pbar.progress(progress, text=f"데이터셋 통계 계산 중... {int(progress*100)}%")
        
    pbar.empty() # 진행률 표시줄을 제거합니다.

    # 누적된 값을 전체 이미지 수로 나누어 최종 평균과 표준편차를 계산합니다.
    mean /= total_images_count
    std /= total_images_count
    
    # 데이터셋의 변환을 원래대로 복구합니다.
    dataset.transform = original_transform
    
    st.success(f"데이터셋 통계 계산 완료: Mean={mean.tolist()}, Std={std.tolist()}")
    
    return mean.tolist(), std.tolist()


# --- Pickle 파일을 위한 커스텀 데이터셋 ---
class PickledDataset(Dataset):
    """
    Pickle 파일에서 로드된 NumPy 배열 형태의 이미지와 레이블을 처리하기 위한
    PyTorch 커스텀 데이터셋 클래스입니다.
    """
    def __init__(self, images, labels, transform=None):
        self.images = images  # 이미지 데이터 (NumPy 배열)
        self.labels = labels  # 레이블 데이터
        self.transform = transform  # 적용할 이미지 변환

    def __len__(self):
        # 데이터셋의 전체 샘플 수를 반환합니다.
        return len(self.images)

    def __getitem__(self, idx):
        # 주어진 인덱스(idx)에 해당하는 이미지와 레이블을 가져옵니다.
        image, label = self.images[idx], self.labels[idx]
        # NumPy 배열을 PIL 이미지로 변환하여 torchvision 변환과 호환되도록 합니다.
        image = Image.fromarray(image)
        # 변환(transform)이 지정된 경우 이미지에 적용합니다.
        if self.transform:
            image = self.transform(image)
        # 이미지와 레이블을 텐서 형태로 반환합니다.
        return image, torch.tensor(label, dtype=torch.long)

# --- 데이터로더 생성 메인 함수 ---
def create_dataloaders(uploaded_file, upload_dir, batch_size, subset_ratio, random_state, normalize_option):
    """
    업로드된 파일을 분석하여 형식에 맞는 데이터로더(DataLoader)를 생성하는 메인 함수입니다.
    - ZIP, TAR.GZ 압축 파일 (ImageFolder, CIFAR-10 형식)
    - Pickle 파일
    """
    if uploaded_file is None: return None, None, None

    # 파일 확장자를 추출합니다.
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    # .tar.gz와 같은 이중 확장자를 처리합니다.
    if uploaded_file.name.endswith('.tar.gz'): file_ext = '.tar.gz'

    # 압축 파일 형식인 경우
    if file_ext in ['.zip', '.tar.gz']:
        extract_path = handle_archive_upload(uploaded_file, upload_dir)
        if not extract_path: return None, None, None
        
        # 데이터셋의 루트 폴더를 찾고, 데이터셋 형식을 감지합니다.
        data_root = find_data_root(extract_path)
        dataset_format = detect_dataset_format(data_root)
        st.info(f"감지된 데이터셋 형식: **{dataset_format}**")

        # 감지된 형식에 따라 적절한 로더 생성 함수를 호출합니다.
        if dataset_format == "ImageFolder":
            return get_loaders_from_image_folder(data_root, batch_size, subset_ratio, random_state, normalize_option)
        elif dataset_format == "CIFAR-10":
            return get_loaders_from_cifar_style(data_root, batch_size, subset_ratio, random_state, normalize_option)
        else:
            st.error("압축 파일 내에서 지원하는 데이터셋 구조(ImageFolder, CIFAR-10)를 찾지 못했습니다.")
            return None, None, None

    # Pickle 파일 형식인 경우
    elif file_ext in ['.pkl', '.pickle']:
        st.info(f"감지된 데이터셋 형식: **Monolithic Pickle**")
        return get_loaders_from_pickle(uploaded_file, upload_dir, batch_size, subset_ratio, random_state, normalize_option)
    # 지원하지 않는 파일 형식인 경우
    else:
        st.error(f"지원하지 않는 파일 형식입니다: {file_ext}")
        return None, None, None

# --- 형식별 로더 생성 함수 ---

def get_loaders_from_image_folder(data_dir, batch_size, subset_ratio, random_state, normalize_option):
    """ImageFolder 구조의 데이터셋으로부터 데이터로더를 생성합니다."""
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        st.error(f"'{train_dir}'와 '{val_dir}' 폴더를 찾을 수 없습니다.")
        return None, None, None

    # 1. 변환(transform) 없이 `ImageFolder` 데이터셋을 우선 로드합니다.
    #    이는 데이터셋 통계 계산 등 원본 데이터 접근이 필요할 때를 위함입니다.
    full_train_dataset = datasets.ImageFolder(train_dir, transform=None)
    
    # 2. 정규화 옵션에 따라 변환(transform)을 결정합니다.
    custom_mean, custom_std = None, None
    if normalize_option == 'dataset':
        # 데이터셋 전체가 클 경우를 대비하여, 일부 서브셋만으로 통계를 계산합니다.
        temp_train_ds, _ = _apply_subset(full_train_dataset, datasets.ImageFolder(val_dir, transform=None), 0.2, random_state)
        st.info(f"정확한 통계 계산을 위해 학습 데이터의 20%({len(temp_train_ds)}개)를 사용합니다.")
        custom_mean, custom_std = calculate_mean_std(temp_train_ds)

    # 계산된 통계(또는 ImageNet 통계)를 사용하여 학습 및 검증 데이터용 변환을 생성합니다.
    train_transform = get_transforms(normalize_option, custom_mean, custom_std)
    val_transform = get_transforms(normalize_option, custom_mean, custom_std)
    
    # 3. 생성된 최종 변환을 데이터셋에 적용합니다.
    full_train_dataset.transform = train_transform
    full_val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    # 4. 지정된 비율(subset_ratio)에 따라 데이터셋의 서브셋을 만들고 데이터로더를 생성합니다.
    train_dataset, val_dataset = _apply_subset(
        full_train_dataset, full_val_dataset, subset_ratio, random_state
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, len(full_train_dataset.classes)

def get_loaders_from_cifar_style(data_dir, batch_size, subset_ratio, random_state, normalize_option):
    """CIFAR-10 스타일의 데이터셋 구조로부터 데이터로더를 생성합니다."""
    cifar_root = data_dir
    # 데이터 폴더 이름이 'cifar-10-batches-py'인 경우, 부모 디렉토리를 root로 사용합니다.
    if os.path.basename(data_dir) in ['cifar-10-batches-py', 'cifar-100-python']:
        cifar_root = os.path.dirname(data_dir)

    try:
        # 1. 변환 없이 CIFAR-10 데이터셋을 로드합니다.
        full_train_dataset = datasets.CIFAR10(root=cifar_root, train=True, download=False, transform=None)
        full_val_dataset = datasets.CIFAR10(root=cifar_root, train=False, download=False, transform=None)
    except RuntimeError as e:
        st.error(f"CIFAR-10 데이터셋 로드에 실패했습니다. 파일이 손상되었거나 구조가 올바르지 않을 수 있습니다.")
        st.info(f"에러: {e}")
        st.info(f"기대한 데이터셋 루트 경로: {cifar_root}")
        return None, None, None

    # 2. 정규화 옵션에 따라 변환을 결정합니다.
    custom_mean, custom_std = None, None
    if normalize_option == 'dataset':
        # CIFAR-10은 크기가 표준화되어 있으므로, 전체 학습셋으로 통계를 계산합니다.
        st.info(f"정확한 통계 계산을 위해 전체 학습 데이터({len(full_train_dataset)}개)를 사용합니다.")
        custom_mean, custom_std = calculate_mean_std(full_train_dataset)

    final_transform = get_transforms(normalize_option, custom_mean, custom_std)

    # 3. 최종 변환을 데이터셋에 적용합니다.
    full_train_dataset.transform = final_transform
    full_val_dataset.transform = final_transform

    # 4. 서브셋을 적용하고 데이터로더를 생성합니다.
    train_dataset, val_dataset = _apply_subset(
        full_train_dataset, full_val_dataset, subset_ratio, random_state
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, len(full_train_dataset.classes)

def get_loaders_from_pickle(uploaded_file, upload_dir, batch_size, subset_ratio, random_state, normalize_option):
    """단일 Pickle 파일로부터 데이터로더를 생성합니다."""
    file_path = os.path.join(upload_dir, uploaded_file.name)
    # 업로드된 파일을 디스크에 저장합니다.
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        # 저장된 Pickle 파일을 읽습니다.
        with open(file_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        st.error(f"Pickle 파일을 읽는 중 오류가 발생했습니다: {e}")
        return None, None, None

    # Pickle 파일에 필수 키들이 있는지 확인합니다.
    required_keys = ['train_images', 'train_labels', 'val_images', 'val_labels', 'classes']
    if not all(key in data for key in required_keys):
        st.error(f"Pickle 파일에 필요한 키가 모두 포함되어 있지 않습니다. ({required_keys})")
        return None, None, None

    # 1. 변환 없이 Pickle 데이터로 커스텀 데이터셋을 생성합니다.
    full_train_dataset = PickledDataset(data['train_images'], data['train_labels'], transform=None)
    full_val_dataset = PickledDataset(data['val_images'], data['val_labels'], transform=None)

    # 2. 정규화 옵션에 따라 변환을 결정합니다.
    custom_mean, custom_std = None, None
    if normalize_option == 'dataset':
        st.info(f"정확한 통계 계산을 위해 전체 학습 데이터({len(full_train_dataset)}개)를 사용합니다.")
        custom_mean, custom_std = calculate_mean_std(full_train_dataset)
    
    final_transform = get_transforms(normalize_option, custom_mean, custom_std)
    
    # 3. 최종 변환을 데이터셋에 적용합니다.
    full_train_dataset.transform = final_transform
    full_val_dataset.transform = final_transform

    # 4. 서브셋을 적용하고 데이터로더를 생성합니다.
    train_dataset, val_dataset = _apply_subset(
        full_train_dataset, full_val_dataset, subset_ratio, random_state
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, len(data['classes'])

# --- 보조 함수 ---

def _apply_subset(train_ds, val_ds, ratio, seed):
    """
    학습 및 검증 데이터셋에 대해 지정된 비율만큼 서브셋 샘플링을 적용합니다.
    `stratify` 옵션을 사용하여 클래스 비율을 유지합니다.
    """
    if ratio < 1.0:
        st.info(f"전체 데이터의 {ratio * 100:.0f}%만 사용하여 학습 및 평가를 진행합니다.")
        
        # 학습 데이터셋 서브셋 추출
        train_targets = train_ds.targets if hasattr(train_ds, 'targets') else [s[1] for s in train_ds.samples]
        train_indices, _ = train_test_split(range(len(train_ds)), train_size=ratio, stratify=train_targets, random_state=seed)
        train_ds = Subset(train_ds, train_indices)
        
        # 검증 데이터셋 서브셋 추출
        val_targets = val_ds.targets if hasattr(val_ds, 'targets') else [s[1] for s in val_ds.samples]
        if len(val_targets) > 0:
            val_indices, _ = train_test_split(range(len(val_ds)), train_size=ratio, stratify=val_targets, random_state=seed)
            val_ds = Subset(val_ds, val_indices)
            
    return train_ds, val_ds

def detect_dataset_format(data_dir):
    """
    주어진 디렉토리의 구조를 분석하여 데이터셋의 형식(CIFAR-10, ImageFolder 등)을 추측합니다.
    """
    dir_content = os.listdir(data_dir)
    # CIFAR-10의 특징적인 파일들을 확인합니다.
    if 'batches.meta' in dir_content and 'data_batch_1' in dir_content:
        return "CIFAR-10"
    # ImageFolder의 특징적인 폴더(train, val)들을 확인합니다.
    if 'train' in dir_content and 'val' in dir_content:
        if os.path.isdir(os.path.join(data_dir, 'train')) and os.path.isdir(os.path.join(data_dir, 'val')):
            return "ImageFolder"
    return "Unknown" # 어느 형식에도 해당하지 않는 경우

def find_data_root(extract_path):
    """
    압축 해제된 폴더 내에서 실제 데이터가 포함된 루트 폴더('train', 'val' 폴더나
    'batches.meta' 파일이 있는 위치)를 탐색합니다.
    """
    for root, dirs, files in os.walk(extract_path):
        if ('train' in dirs and 'val' in dirs) or 'batches.meta' in files:
            return root
    # 찾지 못한 경우, 압축 해제된 최상위 경로를 반환합니다.
    return extract_path

def handle_archive_upload(uploaded_file, upload_dir='uploads'):
    """
    업로드된 압축 파일(ZIP 또는 TAR.GZ)을 서버에 저장하고 지정된 경로에 압축을 해제합니다.
    """
    if not os.path.exists(upload_dir): os.makedirs(upload_dir)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    
    # 파일을 디스크에 씁니다.
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 압축 해제할 폴더 이름을 파일 이름에서 추출합니다.
    extract_base_name = os.path.splitext(os.path.splitext(uploaded_file.name)[0])[0]
    extract_path = os.path.join(upload_dir, extract_base_name)
    if not os.path.exists(extract_path): os.makedirs(extract_path)

    try:
        # 파일 확장자에 따라 다른 압축 해제 방식을 사용합니다.
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