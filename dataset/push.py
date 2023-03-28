import clearml


def main():
   # id, version = clearml.Dataset.get_dataset_id(dataset_project="", dataset_name="")
    dataset = clearml.Dataset.create(dataset_project="deepfake_detection_short_project", dataset_name="deepfake_detection_short_dataset")  # parent_datasets=[id]
    dataset.add_files("./datasets/video/fake")
    dataset.add_files("./datasets/video/real")
    dataset.upload(verbose=True, chunk_size=20)
    dataset.finalize()


if __name__ == "__main__":
    main()