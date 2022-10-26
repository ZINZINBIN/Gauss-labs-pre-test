from src.utils import read_file

DATASET_PATH = "./dataset/data.csv"

if __name__ == "__main__":
    df = read_file(DATASET_PATH, None)
    print(df.head())