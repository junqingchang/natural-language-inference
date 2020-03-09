from utils.dataprocessing import MNLI

TRAIN_DATA_DIR = 'multinli_1.0/multinli_1.0_train.jsonl'

if __name__ == '__main__':
    mnli = MNLI(TRAIN_DATA_DIR)
    print(mnli[0]) # this is the 0th data