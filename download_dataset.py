# download_dataset.py
import kagglehub

def download_dataset():
    print("Downloading MUSDB18 dataset...")
    # Replace with the Kaggle dataset URL or identifier
    path = kagglehub.dataset_download("dhruvpatel1057/musdb18")
    print("Dataset downloaded to:", path)

if __name__ == "__main__":
    download_dataset()




