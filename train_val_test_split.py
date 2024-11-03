import argparse
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import os




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--low_density_regions_dataset', default='/path', type=str, help='path to the low_density_region file')
    parser.add_argument('--high_density_regions_dataset', default='/path', type=str, help='path to the high_density_region file')
    parser.add_argument('--split_low_density_regions', default=0.3, type=float, help='Train/Test split ratio for low_density_regions')
    parser.add_argument('--split_high_density_regions', default=0.1, type=float, help='Train/Test split ratio for low_density_regions')
    parser.add_argument('--mitigating_bias', default=0, type=int, help='mitgating_bias :- True/False')
    parser.add_argument('--folds', default=5, type=int, help='No of folds in K-folds')
    args = parser.parse_args()


    # loading the low_density_region file :-
    df = pd.read_csv(args.low_density_regions_dataset)
    df_low_density_regions = df.sample(frac=1).reset_index(drop=True)

    # loading the high_density_region file :-
    df = pd.read_csv(args.high_density_regions_dataset)
    df_high_density_regions = df.sample(frac=1).reset_index(drop=True)



    if args.mitigating_bias == 0:

        
        
        
        df_low_density_regions_train, df_low_density_regions_test  = train_test_split(df_low_density_regions, test_size = 0.3, stratify=df_low_density_regions['label'], random_state=42)

        df_high_density_regions_train, df_high_density_regions_test  = train_test_split(df_high_density_regions, test_size = 0.1, stratify=df_high_density_regions['label'], random_state=42)

        df_combined_train = pd.concat( [df_low_density_regions_train, df_high_density_regions_train], axis=0).reset_index(drop=True)

        path = '/kaggle/working/'
        os.makedirs(os.path.join(path, 'Test'), exist_ok=True)
        df_low_density_regions_test.to_csv(os.path.join(path, 'Test', 'low_density_regions_test.csv'), index=False)
        df_high_density_regions_test.to_csv(os.path.join(path, 'Test', 'high_density_regions_test.csv'), index=False)



        kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
        i=0
        os.makedirs(os.path.join(path, 'Train_Val_split'), exist_ok=True)

        df_train = df_combined_train
        for train_idx, val_idx in kf.split(df_train['path'], df_train['label']):
            train_df = df_train.iloc[train_idx].reset_index(drop=True)
            train = str(i) + 'train.csv'
            train_df.to_csv(os.path.join(path, 'Train_Val_split', train), index=False)

            df_val = df_train.iloc[val_idx].reset_index(drop=True)
            val = str(i) + 'val.csv'
            df_val.to_csv(os.path.join(path, 'Train_Val_split', val), index=False)
            i = i + 1

        


    
    else:

        

      
        df_low_density_regions_train, df_low_density_regions_test  = train_test_split(df_low_density_regions, test_size = 0.1, stratify=df_low_density_regions['label'], random_state=42)

        df_high_density_regions_train, df_high_density_regions_test  = train_test_split(df_high_density_regions, test_size = 0.1, stratify=df_high_density_regions['label'], random_state=42)

        path = '/kaggle/working/'
        os.makedirs(os.path.join(path, 'Test'), exist_ok=True)
        df_low_density_regions_test = df_low_density_regions_test.drop_duplicates(subset='path').reset_index(drop=True)
        df_low_density_regions_test.to_csv(os.path.join(path, 'Test', 'low_density_regions_test.csv'), index=False)
        df_high_density_regions_test.to_csv(os.path.join(path, 'Test', 'high_density_regions_test.csv'), index=False)


        kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
        



        os.makedirs(os.path.join(path, 'Train_Val_split', 'low_density_regions'), exist_ok=True)
        i=0
        df_train = df_low_density_regions_train
        for train_idx, val_idx in kf.split(df_train['path'], df_train['label']):
            train_df = df_train.iloc[train_idx].reset_index(drop=True)
            train = str(i) + 'train.csv'
            train_df.to_csv(os.path.join(path, 'Train_Val_split', 'low_density_regions', train), index=False)

            df_val = df_train.iloc[val_idx].reset_index(drop=True)
            val = str(i) + 'val.csv'
            df_val.to_csv(os.path.join(path, 'Train_Val_split', 'low_density_regions', val), index=False)
            i = i + 1




        os.makedirs(os.path.join(path, 'Train_Val_split', 'high_density_regions'), exist_ok=True)
        i=0
        df_train = df_high_density_regions_train
        for train_idx, val_idx in kf.split(df_train['path'], df_train['label']):
            train_df = df_train.iloc[train_idx].reset_index(drop=True)
            train = str(i) + 'train.csv'
            train_df.to_csv(os.path.join(path, 'Train_Val_split', 'high_density_regions', train), index=False)

            df_val = df_train.iloc[val_idx].reset_index(drop=True)
            val = str(i) + 'val.csv'
            df_val.to_csv(os.path.join(path, 'Train_Val_split', 'high_density_regions', val), index=False)
            i = i + 1

       

        
        








if __name__ == '__main__':
    main()


