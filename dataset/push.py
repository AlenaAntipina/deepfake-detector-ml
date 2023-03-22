import clearml


def main():
   # id, version = clearml.Dataset.get_dataset_id(dataset_project="", dataset_name="")
    dataset = clearml.Dataset.create(dataset_project="people_deepfake_dataset_project", dataset_name="people_deepfake_dataset")  # parent_datasets=[id]
    dataset.add_files("./dataset/data/Face_only_data")
    dataset.upload(verbose=True, chunk_size=20)
   # dataset.upload(chunk_size=20)
    dataset.finalize()


if __name__ == "__main__":
    main()