import os
import Cooking


if __name__ == '__main__':

    print('=====================================================')
    print('                STARTING PREPARE DATA                ')
    print('=====================================================')

    # Raw data
    RAW_DATA_DIR = 'data_raw/'

    DATA_FOLDERS = ['normal_1', 'normal_2', 'normal_3', 'normal_4', 'normal_5', 'normal_6', 'swerve_1', 'swerve_2', 'swerve_3']

    full_path_raw_folders = [os.path.join(RAW_DATA_DIR, f) for f in DATA_FOLDERS]

    # Training, evaluation and testing splits
    train_eval_test_split = [0.7, 0.2, 0.1]

    # Saves data to h5
    Cooking.cook(full_path_raw_folders, 'data_cooked/', train_eval_test_split)

    print('=====================================================')
    print('              PREPARE DATA COMPLETED                 ')
    print('=====================================================')