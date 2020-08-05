import os
import Cooking


if __name__ == '__main__':

    print('=====================================================')
    print('                STARTING PREPARE DATA                ')
    print('=====================================================')

    # << Point this to the directory containing the raw data >>
    RAW_DATA_DIR = 'data_raw/'

    # << Point this to the desired output directory for the cooked (.h5) data >>
    COOKED_DATA_DIR = 'data_cooked/'

    # The folders to search for data under RAW_DATA_DIR
    DATA_FOLDERS = ['normal_1', 'normal_2', 'normal_3', 'normal_4', 'normal_5', 'normal_6', 'swerve_1', 'swerve_2', 'swerve_3']

    full_path_raw_folders = [os.path.join(RAW_DATA_DIR, f) for f in DATA_FOLDERS]

    train_eval_test_split = [0.7, 0.2, 0.1]
    full_path_raw_folders = [os.path.join(RAW_DATA_DIR, f) for f in DATA_FOLDERS]
    Cooking.cook(full_path_raw_folders, COOKED_DATA_DIR, train_eval_test_split)

    print('=====================================================')
    print('              PREPARE DATA COMPLETED                 ')
    print('=====================================================')