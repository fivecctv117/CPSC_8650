import argparse
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from torch.utils.data import Dataset, DataLoader

# For data preprocess
import numpy as np
import logging
import aecnn, utils

def main(args):
    
    utils.setup_experiment(args)
    utils.init_logging(args)
    
    image_dir = 'data/'
    label_file = args.label_file
    Imgfiles, Scores = utils.load_csv(label_file, args.pt_column)  # 1 represent PT 500, 2 represent PT 4000
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state =42)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = aecnn.cnn_regression().to(device) 
   
    logging.info(f"Built a model consisting of {sum(p.numel() for p in model.parameters()):,} parameters")

    
    num_epochs = 200
    results_all = []
        
    #load autoencoder for latent vectors
    AE_PATH = 'output/autoencoder/cnnreg-Apr-12-10:49:16/checkpoints/checkpoint_best_cnnrg.pt'  
   
    logging.info("10-fold Cross Validation model training and evaluation")
    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(Imgfiles, Scores)):
        logging.info(f'FOLD {fold}')
        img_train, img_test = Imgfiles[train_ids], Imgfiles[test_ids]
        y_train, y_test = Scores[train_ids], Scores[test_ids]
        y_train = [float(s) for s in y_train]
        y_test = [float(s) for s in y_test]
       
        logging.info(img_train)
        logging.info(img_test)
       
        x_train = utils.loadImgs(image_dir, img_train)
        x_test = utils.loadImgs(image_dir, img_test)
        
        train_latentdata = utils.load_latentdata(AE_PATH, x_train)
        test_latentdata = utils.load_latentdata(AE_PATH, x_test)
        
        latent_dataset_pair = torch.utils.data.TensorDataset(torch.tensor(train_latentdata), torch.tensor(y_train))
        if len(img_train)%2 == 0:
            train_loader = torch.utils.data.DataLoader(latent_dataset_pair, batch_size=args.batch_size, num_workers=4, shuffle=True)
        else:
            train_loader = torch.utils.data.DataLoader(latent_dataset_pair, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

        latent_dataset_test = torch.utils.data.TensorDataset(torch.tensor(test_latentdata), torch.tensor(y_test))
        test_loader = torch.utils.data.DataLoader(latent_dataset_test, batch_size=1, num_workers=4, shuffle=False)
        
        model = aecnn.cnn_regression().to(device) 
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # remove weight_decay=1e-5

        len_train = len(img_train)
        len_val = len(img_test)

        pred_best = []
        gt_best = []
        pred_best_test = []
        gt_best_test = []
        min_val_loss = 1000
      
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            pred_list = []
            gt_list = []
            
            model.train()
            for latents, scores in train_loader:
                latents, scores = torch.tensor(latents).cuda(), torch.tensor(scores).cuda()
                optimizer.zero_grad()
               
                pred = model(latents)
                
                loss =  model.loss(pred, scores)
                loss.backward()
                optimizer.step()
              
                pred_list.append(pred)
                gt_list.append(scores)
                running_loss += loss.item() * latents.size(0)
               
            validate_loss = 0
            pred_list_test = []
            gt_list_test = []
            model.eval()
            with torch.no_grad():
                for latents_val, scores_val in test_loader:
                    latents_val, scores_val= torch.tensor(latents_val).cuda(), torch.tensor(scores_val).cuda()
                    out = model(latents_val)
                    curloss = model.loss(out.unsqueeze(0), scores_val)
                    validate_loss += curloss.item() * latents_val.size(0)
                   
                    pred_list_test.append(out)
                    gt_list_test.append(scores_val)
            
            errorlevel = utils.errratelevels(pred_list_test, gt_list_test)
            logging.info(f"Epoch {epoch+1}, Train loss: {running_loss/len_train:0.4}, Val accuracy:{validate_loss/len_val:0.4}, Error level:{errorlevel*1.0/len_val:0.4}")

           
            if min_val_loss > validate_loss/len_val:
                min_val_loss = validate_loss/len_val
                utils.save_checkpoint(epoch+1, model,optimizer,args.checkpoint_dir,fold)
                pred_best = pred_list
                gt_best = gt_list
                pred_best_test = pred_list_test
                gt_best_test = gt_list_test
                
        logging.info("pred_best_test and gt_best_test:")
        logging.info(pred_best_test)
        logging.info(gt_best_test)
        results_all.append((pred_best, gt_best, pred_best_test,  gt_best_test))
        
    utils.results_metrics(results_all)    
         
                
                
def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--label-file", default="PTs_500_4k_blinded.csv", help="path to data directory")
    parser.add_argument("--batch-size", default=8, type=int, help="train batch size")
    parser.add_argument("--model", default="cnnreg", help="model architecture")

    parser.add_argument("--lr", default= 1e-3, type=float, help="learning rate")
    parser.add_argument("--num-epochs", default=200, type=int, help="force stop training at specified epoch")
    parser.add_argument("--num-outputs", default=1, type=int, help="train batch size")
    parser.add_argument("--pt-column", default=2, type=int, help="1 represent PT 500, 2 represent PT 4000")

    parser.add_argument("--output-dir", default="output", help="path to experiment directories")
    parser.add_argument("--experiment", default="cnnreg", help="experiment name to be used with Tensorboard")
    parser.add_argument("--resume-training", action="store_true", help="whether to resume training")
    # Parse twice as model arguments are not known the first time
    args, _ = parser.parse_known_args()
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)