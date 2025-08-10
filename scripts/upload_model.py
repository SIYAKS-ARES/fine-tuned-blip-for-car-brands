from huggingface_hub import create_repo, upload_folder


def main() -> None:
    # Yerel model klasörü ve hedef Hub repo kimliği
    local_model_path = "./car_brand_model_final"
    # Hedef repo (mevcutsa üzerine yazmadan kullanılır), gerekirse değiştirin
    repo_id = "SIYAKSARES/fine-tuned-blip-for-car-brands"

    print(f"Local model path: {local_model_path}")
    print(f"Target Hub repo: {repo_id}")

    # Repo yoksa oluştur; varsa devam et
    print("Ensuring target repository exists (or creating it)...")
    create_repo(repo_id, repo_type="model", exist_ok=True)

    # Klasörü doğrudan Hub'a yükle (Transformer nesnelerini instantiate etmeden)
    print("Uploading local folder to the Hub (this may take several minutes)...")
    upload_folder(
        repo_id=repo_id,
        folder_path=local_model_path,
        path_in_repo=".",
        commit_message="Add fine-tuned BLIP car brand model",
    )

    print("Upload complete! Your model files are now available on the Hub.")


if __name__ == "__main__":
    main()


