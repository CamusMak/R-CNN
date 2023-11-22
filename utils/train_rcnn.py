import sys
sys.path.append("../utils")

from train import train
from validation import validate



def train_model(model,optimizer,train_dataloader,validataion_dataloader,device,n_of_iterations=10,verbose=True):


    """
    
    Trains R-CNN model

    ----------------------------------
    model:
        model to train, in this case R-CNN

    train_dataloader:
        train data 

    validatation_dataloader:
        validataion data

    optimizer:
        optimizer to use to update model parameters
    
    device:
        cuda if available else cpu

    n_of_iterations:
        number of iterations to train model

    verbose:
        if true, print loss during training



    ---------------------------------------------

    iterates over train_dataloder n_of_iterations times

    
    """


    for i in range(n_of_iterations):

        if verbose:

            print("_-"*30,i,"/",n_of_iterations,"-_"*30)

        iteration_train_loss = []

        for ind,batch in enumerate(train_dataloader):

            train_loss = train(model=model,batch=batch,device=device,optimizer=optimizer)
        
            iteration_train_loss.append(train_loss)

            if verbose:
        
                print(f"\tbatch {ind+1} loss: {train_loss}")

        
            
        iteration_validation_loss = []

        for ind,batch in enumerate(validataion_dataloader):

            validation_loss = validate(model=model,batch=batch,device=device,optimizer=optimizer)
            iteration_validation_loss.append(validation_loss)


        mean_train_loss = round(sum(iteration_train_loss)/len(iteration_train_loss),2)
        mean_valid_loss = round(sum(iteration_validation_loss)/len(iteration_validation_loss),2)

        if verbose:
            print(f"{i}/{n_of_iterations}: train loss: {mean_train_loss}, validataion loss: {mean_valid_loss}")

        

            

    