import torch
import nibabel as nib
import skimage.transform as skTrans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

import numpy as np
import os
import csv
from datetime import datetime
import logging
import sys
import math
import cnnreg

def setup_experiment(args):
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)  # for multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.experiment = args.experiment or f"{args.model.replace('_', '-')}"
    if not args.resume_training:
        args.experiment = "-".join([args.experiment, datetime.now().strftime("%b-%d-%H:%M:%S")])

    args.experiment_dir = os.path.join(args.output_dir, args.model, args.experiment)
    os.makedirs(args.experiment_dir, exist_ok=True)

    args.checkpoint_dir = os.path.join(args.experiment_dir, "checkpoints")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    args.log_dir = os.path.join(args.experiment_dir, "logs")
    os.makedirs(args.log_dir, exist_ok=True)
    args.log_file = os.path.join(args.log_dir, "train.log")

def init_logging(args):
    handlers = [logging.StreamHandler()]
    mode = "a" if os.path.isfile(args.resume_training) else "w"
    handlers.append(logging.FileHandler(args.log_file, mode=mode))
    logging.basicConfig(handlers=handlers, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    logging.info("COMMAND: %s" % " ".join(sys.argv))
    logging.info("Arguments: {}".format(vars(args)))
    
def loadImgs(image_dir, imgfiles):
    # Define a list to store image data
    x_train = []
    # Loop through the image files in the directory
    for filename in imgfiles:
        img = nib.load(image_dir + filename.replace("_", "") + '_T1.nii')
        res = skTrans.resize(img.get_fdata(), (112,136,112), order=1, preserve_range=True)
        resized_img = nib.Nifti1Image(res, img.affine).get_fdata()                
        # Append the preprocessed image to the list
        x_train.append(resized_img)
    return np.array(x_train)


def save_checkpoint(step, model, optimizer=None, checkpoint_dir=None, fold=None):
    #model = [model] if model is not None and not isinstance(model, list) else model
    optimizer = [optimizer] if optimizer is not None and not isinstance(optimizer, list) else optimizer
    state_dict = {
            "step": 100,
            "best_step": step,
            "model": model.state_dict(),
            "optimizer": [o.state_dict() for o in optimizer] if optimizer is not None else None
        }
    if fold is not None:
        torch.save(state_dict, os.path.join(checkpoint_dir, f'checkpoint_best_cnnrg{fold}.pt'))
    else:
        torch.save(state_dict, os.path.join(checkpoint_dir, f'checkpoint_best_cnnrg.pt'))


# definition for loading model from a pretrained network file
def load_model(PATH):
    state_dict = torch.load(PATH, map_location="cpu")
    model = cnnreg.cnn_regression()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    model.load_state_dict(state_dict["model"])
    return model, optimizer


# definition for loading model from a pretrained network file
def load_ae(PATH):
    state_dict = torch.load(PATH, map_location="cpu") #, map_location="cpu"
    model = cnnreg.autoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    model.load_state_dict(state_dict["model"])
    return model

def load_csv(label_path, pt_column):
    with open(label_path, 'r') as fp:
        data = list(csv.reader(fp))
        data = np.array(data[1:])
    imgfiles = data[:, 0]
    scores = data[:, pt_column]  
    return imgfiles, scores

def save_csvfile(xtest, ytest, path, filename, pt_type):
    # Combine the two columns into a list of tuples
    data = list(zip(xtest, ytest))
    file_name = os.path.join(path,filename + '.csv')
    # Write data to the CSV file
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', pt_type])  # Write header
        writer.writerows(data)  # Write data
        
def load_latentdata(PATH, x_imgs, aug = False):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    model = load_ae(PATH, aug)
    model = model.to(device) 

    model.eval()
    data_loader = torch.utils.data.DataLoader(x_imgs, batch_size=1, num_workers=5, shuffle=False)
    latent_list = []
    with torch.no_grad():
        for input in data_loader:
            latents, outputs = model(input.float().cuda())
            latent_list.append(latents[0].cpu().detach().numpy())
    return np.array(latent_list)


def results_metrics(results_all):
    train_metrics = []
    test_metrics = []
    rlen = len(results_all)
    for i in range(0,rlen):
        pred_score = [round(element.item(), 2) for row in results_all[i][0] for element in row]
        gr_score = [element.item() for row in results_all[i][1] for element in row]
        
        mse = mean_squared_error(gr_score, pred_score)
        rmse = math.sqrt(mse)
        # R-squared Score
        r2 = r2_score(gr_score, pred_score)
        corr, _ = pearsonr(pred_score, gr_score)
        
        pred_score_test = [round(element.item(), 2) for element in results_all[i][2]]
        gr_score_test = [element.item() for element in results_all[i][3]]
        
        mse_test = mean_squared_error(gr_score_test, pred_score_test)
        rmse_test = math.sqrt(mse_test)
        # R-squared Score
        r2_test = r2_score(gr_score_test, pred_score_test)
        corr_test, _ = pearsonr(pred_score_test, gr_score_test)
        
        logging.info(f"Fold {i}")
        logging.info(f"Root Mean Squared Error   Trainset: {rmse:0.4}   Testset:{rmse_test:0.4}")
        logging.info(f"R-squared                 Trainset: {r2:0.4}   Testset:{r2_test:0.4}")
        logging.info(f"Pearson Correlation       Trainset: {corr:0.4}   Testset:{corr_test:0.4}")
        train_metrics.append((mse, r2, corr))
        test_metrics.append((mse_test, r2_test, corr_test))
        
    logging.info("============Average Performance on Testset from 5-Fold Stratified Cross Validation============ ")
    # Calculate the average of each element
    train_metrics_avg = [sum(x) / rlen for x in zip(*train_metrics)]
    test_metrics_avg = [sum(x) / rlen for x in zip(*test_metrics)]
    test_rmse_avg = math.sqrt(test_metrics_avg[0])
    
    logging.info(f"Root Mean Squared Error   Testset: {test_rmse_avg:0.5}")
    logging.info(f"R-squared                 Testset: {test_metrics_avg[1]:0.5}")
    logging.info(f"Pearson Correlation       Testset: {test_metrics_avg[2]:0.5}")
    