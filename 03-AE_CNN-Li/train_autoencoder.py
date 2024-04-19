import argparse
import sys
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import Dataset, DataLoader
# For data preprocess
import numpy as np

import csv
import os
import logging
import aecnn, utils

def main(args):
    
    utils.setup_experiment(args)
    utils.init_logging(args)
    
    image_dir = 'data/'
    label_file = 'PTs_500_4k_blinded.csv'
   
    Imgfiles, Scores = utils.load_csv(label_file, args.pt_column)  # 1 represent PT 500, 2 represent PT 4000

    img_train, img_test, y_train, y_test = train_test_split(Imgfiles, Scores, test_size=0.2)

    x_train = utils.loadImgs(image_dir, img_train, False, True)
    x_test = utils.loadImgs(image_dir, img_test, False, True)
    
    y_train = [float(s) for s in y_train]

    train_loader = torch.utils.data.DataLoader(x_train, batch_size=args.batch_size, num_workers=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(x_test, batch_size=1, num_workers=4, shuffle=False)


    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = aecnn.autoencoder().to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # remove weight_decay=1e-5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=0.001, eta_min=1e-5)
    
    logging.info(f"Built a model consisting of {sum(p.numel() for p in model.parameters()):,} parameters")

    len_train = len(y_train) 
    len_test = len(y_test) 
    num_epochs = 500
    min_loss = 1000

    for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for _,inputs in enumerate(train_loader, 0):
                inputs = inputs.type(torch.FloatTensor).cuda()
                optimizer.zero_grad()
                latents,outputs = model(inputs)
                loss = model.loss(outputs.cuda(), inputs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            scheduler.step()
            
            tesing_loss = 0
            model.eval()
            with torch.no_grad():
                for _,imgs in enumerate(test_loader, 0):
                    imgs = imgs.type(torch.FloatTensor).cuda()
                    _, out = model(imgs)
                    testloss = model.loss(out.cuda(), imgs)
                    tesing_loss += testloss.item() * imgs.size(0)
            logging.info(f"Epoch {epoch+1}, Train loss: {running_loss/len_train:0.4}, Val accuracy:{tesing_loss/len_test:0.4}")
            if min_loss > tesing_loss:
                min_loss = tesing_loss
                utils.save_checkpoint(epoch+1, model,optimizer,args.checkpoint_dir)
                
                
                
def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--batch-size", default=8, type=int, help="train batch size")
    parser.add_argument("--model", default="autoencoder", help="model architecture")

    # Add optimization arguments
    parser.add_argument("--lr", default= 1e-3, type=float, help="learning rate")
    parser.add_argument("--num-epochs", default=8, type=int, help="force stop training at specified epoch")
    
    #Add output arguments
    parser.add_argument("--num-outputs", default=1, type=int, help="train batch size")
    parser.add_argument("--pt-column", default=1, type=int, help="1 represent PT 500, 2 represent PT 4000")

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