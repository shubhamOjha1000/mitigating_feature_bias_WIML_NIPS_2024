import pandas as pd
import numpy as np
import os
import argparse
from utils import calculating_prob_mass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='/path', type=str, help='path to the feature file')
    parser.add_argument('--mitigating_bias', default=0, type=int, help='mitgating_bias :- True/False')
    args = parser.parse_args()

    # loading the feature file :-
    df = pd.read_csv(args.dataset)
    df = df.sample(frac=1).reset_index(drop=True)

    
    # For class 0:-
    df_concat_low_density_regions_class_0, df_concat_high_density_regions_class_0 = calculating_prob_mass(df, 0)
    

    # For class 1:-
    df_concat_low_density_regions_class_1, df_concat_high_density_regions_class_1 = calculating_prob_mass(df, 1)
   

    # For class 2:-
    df_concat_low_density_regions_class_2, df_concat_high_density_regions_class_2 = calculating_prob_mass(df, 2)
    

    # For class 3:-
    df_concat_low_density_regions_class_3, df_concat_high_density_regions_class_3 = calculating_prob_mass(df, 3)
    

    # For class 4:-
    df_concat_low_density_regions_class_4, df_concat_high_density_regions_class_4 = calculating_prob_mass(df, 4)
    

    # For class 5:-
    df_concat_low_density_regions_class_5, df_concat_high_density_regions_class_5 = calculating_prob_mass(df, 5)
    

    # For class 6:-
    df_concat_low_density_regions_class_6, df_concat_high_density_regions_class_6 = calculating_prob_mass(df, 6)
    

    # For class 7:-
    df_concat_low_density_regions_class_7, df_concat_high_density_regions_class_7 = calculating_prob_mass(df, 7)




    path = '/kaggle/working/'




    if args.mitigating_bias == 0:
        """
        
        os.makedirs(os.path.join(path, 'class_0'), exist_ok=True)
        df_concat_low_density_regions_class_0.to_csv(os.path.join(path, 'class_0', 'low_density_regions.csv'), index=False)
        df_concat_high_density_regions_class_0.to_csv(os.path.join(path, 'class_0', 'high_density_regions.csv'), index=False)

        os.makedirs(os.path.join(path, 'class_1'), exist_ok=True)
        df_concat_low_density_regions_class_1.to_csv(os.path.join(path, 'class_1', 'low_density_regions.csv'), index=False)
        df_concat_high_density_regions_class_1.to_csv(os.path.join(path, 'class_1', 'high_density_regions.csv'), index=False)

        os.makedirs(os.path.join(path, 'class_2'), exist_ok=True)
        df_concat_low_density_regions_class_2.to_csv(os.path.join(path, 'class_2', 'low_density_regions.csv'), index=False)
        df_concat_high_density_regions_class_2.to_csv(os.path.join(path, 'class_2', 'high_density_regions.csv'), index=False)

        os.makedirs(os.path.join(path, 'class_3'), exist_ok=True)
        df_concat_low_density_regions_class_3.to_csv(os.path.join(path, 'class_3', 'low_density_regions.csv'), index=False)
        df_concat_high_density_regions_class_3.to_csv(os.path.join(path, 'class_3', 'high_density_regions.csv'), index=False)

        os.makedirs(os.path.join(path, 'class_4'), exist_ok=True)
        df_concat_low_density_regions_class_4.to_csv(os.path.join(path, 'class_4', 'low_density_regions.csv'), index=False)
        df_concat_high_density_regions_class_4.to_csv(os.path.join(path, 'class_4', 'high_density_regions.csv'), index=False)

        os.makedirs(os.path.join(path, 'class_5'), exist_ok=True)
        df_concat_low_density_regions_class_5.to_csv(os.path.join(path, 'class_5', 'low_density_regions.csv'), index=False)
        df_concat_high_density_regions_class_5.to_csv(os.path.join(path, 'class_5', 'high_density_regions.csv'), index=False)

        os.makedirs(os.path.join(path, 'class_6'), exist_ok=True)
        df_concat_low_density_regions_class_6.to_csv(os.path.join(path, 'class_6', 'low_density_regions.csv'), index=False)
        df_concat_high_density_regions_class_6.to_csv(os.path.join(path, 'class_6', 'high_density_regions.csv'), index=False)

        os.makedirs(os.path.join(path, 'class_7'), exist_ok=True)
        df_concat_low_density_regions_class_7.to_csv(os.path.join(path, 'class_7', 'low_density_regions.csv'), index=False)
        df_concat_high_density_regions_class_7.to_csv(os.path.join(path, 'class_7', 'high_density_regions.csv'), index=False)
        """



        df_combined_low_density_regions = pd.concat([df_concat_low_density_regions_class_0, df_concat_low_density_regions_class_1, df_concat_low_density_regions_class_2, df_concat_low_density_regions_class_3, df_concat_low_density_regions_class_4, df_concat_low_density_regions_class_5, df_concat_low_density_regions_class_6, df_concat_low_density_regions_class_7], axis=0).reset_index(drop=True)
        df_combined_low_density_regions = df_combined_low_density_regions.sample(frac=1).reset_index(drop=True)
        df_combined_low_density_regions.to_csv(os.path.join(path, 'low_density_regions.csv'), index=False)

        df_combined_high_density_regions = pd.concat([df_concat_high_density_regions_class_0, df_concat_high_density_regions_class_1, df_concat_high_density_regions_class_2, df_concat_high_density_regions_class_3, df_concat_high_density_regions_class_4, df_concat_high_density_regions_class_5, df_concat_high_density_regions_class_6, df_concat_high_density_regions_class_7], axis=0).reset_index(drop=True)
        df_combined_high_density_regions = df_combined_high_density_regions.sample(frac=1).reset_index(drop=True)
        df_combined_high_density_regions.to_csv(os.path.join(path, 'high_density_regions.csv'), index=False)



    else:

        low_density_regions_class_0 = df_concat_high_density_regions_class_0.shape[0]
        df_low_density_regions_class_0_oversampled = pd.concat([df_concat_low_density_regions_class_0] * (low_density_regions_class_0 // len(df_concat_low_density_regions_class_0) + 1), ignore_index=True).iloc[:low_density_regions_class_0]
        """
        os.makedirs(os.path.join(path, 'class_0'), exist_ok=True)
        df_low_density_regions_class_0_oversampled.to_csv(os.path.join(path, 'class_0', 'low_density_regions.csv'), index=False)
        df_concat_high_density_regions_class_0.to_csv(os.path.join(path, 'class_0', 'high_density_regions.csv'), index=False)
        """


        low_density_regions_class_1 = df_concat_high_density_regions_class_1.shape[0]
        df_low_density_regions_class_1_oversampled = pd.concat( [df_concat_low_density_regions_class_1] * (low_density_regions_class_1 // len(df_concat_low_density_regions_class_1) + 1 ), ignore_index=True).iloc[:low_density_regions_class_1]
        """
        os.makedirs(os.path.join(path, 'class_1'), exist_ok=True)
        df_low_density_regions_class_1_oversampled.to_csv(os.path.join(path, 'class_1', 'low_density_regions.csv'), index=False)
        df_concat_high_density_regions_class_1.to_csv(os.path.join(path, 'class_1', 'high_density_regions.csv'), index=False)
        """


        low_density_regions_class_2 = df_concat_high_density_regions_class_2.shape[0]
        df_low_density_regions_class_2_oversampled = pd.concat( [df_concat_low_density_regions_class_2] * (low_density_regions_class_2 // len(df_concat_low_density_regions_class_2) + 1 ), ignore_index=True).iloc[:low_density_regions_class_2]
        """
        os.makedirs(os.path.join(path, 'class_2'), exist_ok=True)
        df_low_density_regions_class_2_oversampled.to_csv(os.path.join(path, 'class_2', 'low_density_regions.csv'), index=False)
        df_concat_high_density_regions_class_2.to_csv(os.path.join(path, 'class_2', 'high_density_regions.csv'), index=False)
        """

        low_density_regions_class_3 = df_concat_high_density_regions_class_3.shape[0]
        df_low_density_regions_class_3_oversampled = pd.concat( [df_concat_low_density_regions_class_3] * (low_density_regions_class_3 // len(df_concat_low_density_regions_class_3) + 1 ), ignore_index=True).iloc[:low_density_regions_class_3]
        """
        os.makedirs(os.path.join(path, 'class_3'), exist_ok=True)
        df_low_density_regions_class_3_oversampled.to_csv(os.path.join(path, 'class_3', 'low_density_regions.csv'), index=False)
        df_concat_high_density_regions_class_3.to_csv(os.path.join(path, 'class_3', 'high_density_regions.csv'), index=False)
        """

        low_density_regions_class_4 = df_concat_high_density_regions_class_4.shape[0]
        df_low_density_regions_class_4_oversampled = pd.concat( [df_concat_low_density_regions_class_4] * (low_density_regions_class_4 // len(df_concat_low_density_regions_class_4) + 1 ), ignore_index=True).iloc[:low_density_regions_class_4]
        """
        os.makedirs(os.path.join(path, 'class_4'), exist_ok=True)
        df_low_density_regions_class_4_oversampled.to_csv(os.path.join(path, 'class_4', 'low_density_regions.csv'), index=False)
        df_concat_high_density_regions_class_4.to_csv(os.path.join(path, 'class_4', 'high_density_regions.csv'), index=False)
        """

        low_density_regions_class_5 = df_concat_high_density_regions_class_5.shape[0]
        df_low_density_regions_class_5_oversampled = pd.concat( [df_concat_low_density_regions_class_5] * (low_density_regions_class_5 // len(df_concat_low_density_regions_class_5) + 1 ), ignore_index=True).iloc[:low_density_regions_class_5]
        """
        os.makedirs(os.path.join(path, 'class_5'), exist_ok=True)
        df_low_density_regions_class_5_oversampled.to_csv(os.path.join(path, 'class_5', 'low_density_regions.csv'), index=False)
        df_concat_high_density_regions_class_5.to_csv(os.path.join(path, 'class_5', 'high_density_regions.csv'), index=False)
        """

        low_density_regions_class_6 = df_concat_high_density_regions_class_6.shape[0]
        df_low_density_regions_class_6_oversampled = pd.concat( [df_concat_low_density_regions_class_6] * (low_density_regions_class_6 // len(df_concat_low_density_regions_class_6) + 1 ), ignore_index=True).iloc[:low_density_regions_class_6]
        """
        os.makedirs(os.path.join(path, 'class_6'), exist_ok=True)
        df_low_density_regions_class_6_oversampled.to_csv(os.path.join(path, 'class_6', 'low_density_regions.csv'), index=False)
        df_concat_high_density_regions_class_6.to_csv(os.path.join(path, 'class_6', 'high_density_regions.csv'), index=False)
        """

        low_density_regions_class_7 = df_concat_high_density_regions_class_7.shape[0]
        df_low_density_regions_class_7_oversampled = pd.concat( [df_concat_low_density_regions_class_7] * (low_density_regions_class_7 // len(df_concat_low_density_regions_class_7) + 1 ), ignore_index=True).iloc[:low_density_regions_class_7]
        """
        os.makedirs(os.path.join(path, 'class_7'), exist_ok=True)
        df_low_density_regions_class_7_oversampled.to_csv(os.path.join(path, 'class_7', 'low_density_regions.csv'), index=False)
        df_concat_high_density_regions_class_7.to_csv(os.path.join(path, 'class_7', 'high_density_regions.csv'), index=False)
        """

        
        df_combined_low_density_regions = pd.concat([df_low_density_regions_class_0_oversampled, df_low_density_regions_class_1_oversampled, df_low_density_regions_class_2_oversampled, df_low_density_regions_class_3_oversampled, df_low_density_regions_class_4_oversampled, df_low_density_regions_class_5_oversampled, df_low_density_regions_class_6_oversampled, df_low_density_regions_class_7_oversampled,], axis=0).reset_index(drop=True)
        df_combined_low_density_regions = df_combined_low_density_regions.sample(frac=1).reset_index(drop=True)
        df_combined_low_density_regions.to_csv(os.path.join(path, 'low_density_regions.csv'), index=False)
        df_combined_high_density_regions = pd.concat([df_concat_high_density_regions_class_0, df_concat_high_density_regions_class_1, df_concat_high_density_regions_class_2, df_concat_high_density_regions_class_3, df_concat_high_density_regions_class_4, df_concat_high_density_regions_class_5, df_concat_high_density_regions_class_6, df_concat_high_density_regions_class_7], axis=0).reset_index(drop=True)
        df_combined_high_density_regions = df_combined_high_density_regions.sample(frac=1).reset_index(drop=True)
        df_combined_high_density_regions.to_csv(os.path.join(path, 'high_density_regions.csv'), index=False)





if __name__ == '__main__':
    main()

