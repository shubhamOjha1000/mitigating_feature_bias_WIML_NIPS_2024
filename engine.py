import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os


def train(data_loader, model, optimizer, mitigating_bias, train_loader_low_density_regions, train_loader_high_density_regions):

    # put the model in train mode
    model.train()

    if mitigating_bias == 0:

        for data in data_loader:
            feature = data[0].float()
            label = data[1]

            # Check if CUDA is available
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            # Move the tensor to the selected device (CPU or CUDA)
            feature = feature.to(device)
            label = label.to(device)

            outputs = model(feature)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, label)

            # zero grad the optimizer
            optimizer.zero_grad()

            # calculate the gradient
            loss.backward()

            # update the weights
            optimizer.step()

            torch.cuda.empty_cache()
            



    else:

        for data_low_density_regions, data_high_density_regions in zip(train_loader_low_density_regions, train_loader_high_density_regions):

            feature_low_density_regions = data_low_density_regions[0].float()
            label_low_density_regions = data_low_density_regions[1]

            feature_high_density_regions = data_high_density_regions[0].float()
            label_high_density_regions = data_high_density_regions[1]

            feature = torch.cat((feature_low_density_regions, feature_high_density_regions), dim=0)
            label = torch.cat((label_low_density_regions, label_high_density_regions), dim=0)

            # Check if CUDA is available
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            # Move the tensor to the selected device (CPU or CUDA)
            feature = feature.to(device)
            label = label.to(device)

            outputs = model(feature)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, label)

            # zero grad the optimizer
            optimizer.zero_grad()

            # calculate the gradient
            loss.backward()

            # update the weights
            optimizer.step()

            torch.cuda.empty_cache()



            """
            
            print(f'feature_low_density_regions.shape :- {feature_low_density_regions.shape}')
            print(f'feature_high_density_regions.shape :- {feature_high_density_regions.shape}')

            print(f'label_low_density_regions.shape :- {label_low_density_regions.shape}')
            print(f'label_high_density_regions.shape :- {label_high_density_regions.shape}')
            """


        


    




def val(data_loader, model, mitigating_bias, val_loader_low_density_regions, val_loader_high_density_regions):
    val_loss_list = []
    final_output = []
    final_label = []

    # put model in evaluation mode
    model.eval()

    if mitigating_bias == 0:
        with torch.no_grad():
            for data in data_loader:
                feature = data[0].float()
                label = data[1]

                # Check if CUDA is available
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                else:
                    device = torch.device("cpu")

                # Move the tensor to the selected device (CPU or CUDA)
                feature = feature.to(device)
                label = label.to(device)

                outputs = model(feature)

                criterion = nn.CrossEntropyLoss()
                temp_val_loss = criterion(outputs, label)
                val_loss_list.append(temp_val_loss)
                softmax_values = F.softmax(outputs, dim=1)
                outputs = torch.argmax(softmax_values, dim=1).int()

                OUTPUTS = outputs.detach().cpu().tolist()
                final_output.extend(OUTPUTS)
                final_label.extend(label.detach().cpu().tolist())

                torch.cuda.empty_cache()


    else:
        for data_low_density_regions, data_high_density_regions in zip(val_loader_low_density_regions , val_loader_high_density_regions):

            feature_low_density_regions = data_low_density_regions[0].float()
            label_low_density_regions = data_low_density_regions[1]

            feature_high_density_regions =  data_high_density_regions[0].float()
            label_high_density_regions =  data_high_density_regions[1]

            feature = torch.cat((feature_low_density_regions, feature_high_density_regions), dim=0)
            label = torch.cat((label_low_density_regions, label_high_density_regions), dim=0)

            # Check if CUDA is available
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            # Move the tensor to the selected device (CPU or CUDA)
            feature = feature.to(device)
            label = label.to(device)

            outputs = model(feature)

            criterion = nn.CrossEntropyLoss()
            temp_val_loss = criterion(outputs, label)
            val_loss_list.append(temp_val_loss)
            softmax_values = F.softmax(outputs, dim=1)
            outputs = torch.argmax(softmax_values, dim=1).int()

            OUTPUTS = outputs.detach().cpu().tolist()
            final_output.extend(OUTPUTS)
            final_label.extend(label.detach().cpu().tolist())

            torch.cuda.empty_cache()


            """
            
            print(f'feature_low_density_regions.shape :- {feature_low_density_regions.shape}')
            print(f'feature_high_density_regions.shape :- {feature_high_density_regions.shape}')

            print(f'label_low_density_regions.shape :- {label_low_density_regions.shape}')
            print(f'label_high_density_regions.shape :- {label_high_density_regions.shape}')
            """



    return final_output, final_label, sum(val_loss_list)/len(val_loss_list)






def test(test_loader_low_density_region, test_loader_high_density_regions, model):
    final_output_low_density_region = []
    final_label_low_density_region = []

    final_output_high_density_regions = []
    final_label_high_density_regions = []
    

    # put model in evaluation mode
    model.eval()
    with torch.no_grad():

        for data in test_loader_low_density_region:
            feature = data[0].float()
            label = data[1]

            # Check if CUDA is available
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

             # Move the tensor to the selected device (CPU or CUDA)
            feature = feature.to(device)
            label = label.to(device)

            outputs = model(feature)
            softmax_values = F.softmax(outputs, dim=1)
            outputs = torch.argmax(softmax_values, dim=1).int()
            OUTPUTS = outputs.detach().cpu().tolist()

            final_output_low_density_region.extend(OUTPUTS)
            final_label_low_density_region.extend(label.detach().cpu().tolist())

            torch.cuda.empty_cache()

        

        for data in test_loader_high_density_regions:
            feature = data[0].float()
            label = data[1]

            # Check if CUDA is available
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

             # Move the tensor to the selected device (CPU or CUDA)
            feature = feature.to(device)
            label = label.to(device)

            outputs = model(feature)
            softmax_values = F.softmax(outputs, dim=1)
            outputs = torch.argmax(softmax_values, dim=1).int()
            OUTPUTS = outputs.detach().cpu().tolist()

            final_output_high_density_regions.extend(OUTPUTS)
            final_label_high_density_regions.extend(label.detach().cpu().tolist())

            torch.cuda.empty_cache()


    return final_output_low_density_region, final_label_low_density_region, final_output_high_density_regions, final_label_high_density_regions






