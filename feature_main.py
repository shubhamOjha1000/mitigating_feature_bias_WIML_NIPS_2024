import argparse
import os
from dataset import feature_dataset
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import engine 
from utils import Multiclass_classification_metrices
from model import Feature_FC_layer_for_Diet



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', default=8, type=int, help='  CRIC:- 8 classes')
    parser.add_argument('--num_epochs', default=100, type=int, help= 'Number of total training epochs')
    parser.add_argument('--model', default='Feature_FC_layer_for_Diet', type=str, help='Model to be used')
    parser.add_argument('--batch_size', default=512, type=int, help='Batch size for the dataloader')
    parser.add_argument('--batch_size_low', default=512, type=int, help='Batch size for the dataloader')
    parser.add_argument('--batch_size_high', default=512, type=int, help='Batch size for the dataloader')
    parser.add_argument('--num_workers', default=4, type=int, help='num workers for dataloader')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight Decay')
    parser.add_argument('--patience', default=20, type=int, help='Representing the number of consecutive epochs where the performance metric does not improve before training stops')
    parser.add_argument('--folds', default=5, type=int, help='No of folds in K-folds')
    parser.add_argument('--mitigating_bias', default=0, type=int, help='mitgating_bias :- True/False')
    args = parser.parse_args()



    path = '/kaggle/working/' 


    for fold in range(args.folds):
        print(f'fold :- {fold}')

        if args.mitigating_bias == 0:

            train = str(fold) + 'train.csv'
            train_path = os.path.join(path, 'Train_Val_split', train)
            df_train = pd.read_csv(train_path)
            df_train = df_train.sample(frac=1).reset_index(drop=True)


            train_dataset = feature_dataset(df_train)
            train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)



            val = str(fold) + 'val.csv'
            val_path = os.path.join(path, 'Train_Val_split', val)
            df_val = pd.read_csv(val_path)
            df_val = df_val.sample(frac=1).reset_index(drop=True)

            val_dataset = feature_dataset(df_val)
            val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)




            test_path_low_density_regions = os.path.join(path, 'Test', 'low_density_regions_test.csv')
            df_test_low_density_regions = pd.read_csv(test_path_low_density_regions)
            df_test_low_density_regions = df_test_low_density_regions.sample(frac=1).reset_index(drop=True)

            test_dataset_low_density_regions = feature_dataset(df_test_low_density_regions)
            test_loader_low_density_region = DataLoader(test_dataset_low_density_regions, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)


            test_path_high_density_regions = os.path.join(path, 'Test', 'high_density_regions_test.csv')
            df_test_high_density_regions = pd.read_csv(test_path_high_density_regions)
            df_test_high_density_regions = df_test_high_density_regions.sample(frac=1).reset_index(drop=True)


            test_dataset_high_density_regions = feature_dataset(df_test_high_density_regions)
            test_loader_high_density_regions = DataLoader(test_dataset_high_density_regions, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)





        else:

            train = str(fold) + 'train.csv'

            train_path_low_density_regions = os.path.join(path, 'Train_Val_split', 'low_density_regions', train)
            df_train_low_density_regions = pd.read_csv(train_path_low_density_regions)
            df_train_low_density_regions = df_train_low_density_regions.sample(frac=1).reset_index(drop=True)

            train_dataset_low_density_regions = feature_dataset(df_train_low_density_regions)
            train_loader_low_density_regions = DataLoader(train_dataset_low_density_regions, batch_size = args.batch_size_low, shuffle = True, num_workers = args.num_workers)




            train_path_high_density_regions = os.path.join(path, 'Train_Val_split', 'high_density_regions', train)
            df_train_high_density_regions = pd.read_csv(train_path_high_density_regions)
            df_train_high_density_regions = df_train_high_density_regions.sample(frac=1).reset_index(drop=True)

            train_dataset_high_density_regions = feature_dataset(df_train_high_density_regions)
            train_loader_high_density_regions= DataLoader(train_dataset_high_density_regions, batch_size = args.batch_size_high, shuffle = True, num_workers = args.num_workers)




            val = str(fold) + 'val.csv'

            val_path_low_density_regions = os.path.join(path, 'Train_Val_split', 'low_density_regions', val)
            df_val_low_density_regions = pd.read_csv(val_path_low_density_regions)
            df_val_low_density_regions = df_val_low_density_regions.drop_duplicates(subset='path').reset_index(drop=True)
            df_val_low_density_regions = df_val_low_density_regions.sample(frac=1).reset_index(drop=True)

            val_dataset_low_density_regions = feature_dataset(df_val_low_density_regions)
            val_loader_low_density_regions = DataLoader(val_dataset_low_density_regions, batch_size = args.batch_size_low, shuffle = True, num_workers = args.num_workers)


            val_path_high_density_regions = os.path.join(path, 'Train_Val_split', 'high_density_regions', val)
            df_val_high_density_regions = pd.read_csv(val_path_high_density_regions)
            df_val_high_density_regions = df_val_high_density_regions.sample(frac=1).reset_index(drop=True)

            val_dataset_high_density_regions = feature_dataset(df_val_high_density_regions)
            val_loader_high_density_regions = DataLoader(val_dataset_high_density_regions, batch_size = args.batch_size_high, shuffle = True, num_workers = args.num_workers)





            test_path_low_density_regions = os.path.join(path, 'Test', 'low_density_regions_test.csv')
            df_test_low_density_regions = pd.read_csv(test_path_low_density_regions)
            df_test_low_density_regions = df_test_low_density_regions.sample(frac=1).reset_index(drop=True)

            test_dataset_low_density_regions = feature_dataset(df_test_low_density_regions)
            test_loader_low_density_region = DataLoader(test_dataset_low_density_regions, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)


            test_path_high_density_regions = os.path.join(path, 'Test', 'high_density_regions_test.csv')
            df_test_high_density_regions = pd.read_csv(test_path_high_density_regions)
            df_test_high_density_regions = df_test_high_density_regions.sample(frac=1).reset_index(drop=True)


            test_dataset_high_density_regions = feature_dataset(df_test_high_density_regions)
            test_loader_high_density_regions = DataLoader(test_dataset_high_density_regions, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)













        # model :- 
        if args.model == 'Feature_FC_layer_for_Diet':
            model = Feature_FC_layer_for_Diet()
            device = torch.device("cuda")
            model.to(device)


        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.99))

        patience = args.patience
        best_val_loss = np.inf
        epochs_without_improvement = 0
        best_model_state = None


        for epoch in range(args.num_epochs):
            print(f'epoch :-{epoch}')

            train_loader = 1
            engine.train(train_loader, model, optimizer, args.mitigating_bias, train_loader_low_density_regions, train_loader_high_density_regions)

            val_loader = 1
            _, _, val_loss= engine.val(val_loader, model, args.mitigating_bias, val_loader_low_density_regions, val_loader_high_density_regions)
            print(f'val_loss:-{val_loss}')

            # early stopping :- 
            if epoch>=10:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    # Save the state dictionary of the best model
                    best_model_state = model.state_dict()

                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f"Early stopping after {epoch+1} epochs without improvement.")
                        break

        
        # Load the best model state dictionary for val metrices :-
        if best_model_state is not None:
            model.load_state_dict(best_model_state)


        print()
        print('val')
        val_predictions, val_labels, _ = engine.val(val_loader, model, args.mitigating_bias, val_loader_low_density_regions, val_loader_high_density_regions)
        val_auc, val_acc = Multiclass_classification_metrices(val_labels, val_predictions, args.num_classes)
        print(f'val_auc:-{val_auc}')
        print(f'val_acc:-{val_acc}')
        print()

        
        print()
        print('test')
        predictions_low_density_region, label_low_density_region, predictions_high_density_regions, label_high_density_regions = engine.test(test_loader_low_density_region, test_loader_high_density_regions, model)


        test_auc_low_density_region, test_acc_low_density_region = Multiclass_classification_metrices(label_low_density_region, predictions_low_density_region, args.num_classes)
        print(f'test_auc_low_density_region:-{test_auc_low_density_region}')
        print(f'test_acc_low_density_region:-{test_acc_low_density_region}')
        print()


        test_auc_high_density_regions, test_acc_high_density_regions = Multiclass_classification_metrices(label_high_density_regions, predictions_high_density_regions, args.num_classes)
        print(f'test_auc_high_density_regions:-{test_auc_high_density_regions}')
        print(f'test_acc_high_density_regions:-{test_acc_high_density_regions}')
        print()








if __name__ == '__main__':
    main()