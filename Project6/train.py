import torch
import sys
import numpy as np
import itertools
from models import *
from dataset import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
import time
import datetime
import copy
import random
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/scratch/user/lihao/videos", help="Path to dataset")
    parser.add_argument("--class_path", type=str, default="/home/lihao/LSTMF/classes.txt", help="Path to frames")
    parser.add_argument("--num_epochs", type=int, default=21, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=12, help="Size of each training batch")
    parser.add_argument("--sequence_length", type=int, default=40, help="Number of frames in each sequence")
    parser.add_argument("--img_dim", type=int, default=224, help="Height / width dimension")
    parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
    parser.add_argument("--num_joint", type=int, default=20, help="Number of pose joints")
    parser.add_argument("--latent_dim", type=int, default=512, help="Dimensionality of the latent representation")
    parser.add_argument("--checkpoint_model", type=str, default="", help="Optional path to checkpoint model")
    parser.add_argument(
        "--checkpoint_interval", type=int, default=5, help="Interval between saving model checkpoints"
    )
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    image_shape = (opt.channels, opt.img_dim, opt.img_dim)

    # Define training set
    train_dataset = Dataset(
        dataset_path=opt.dataset_path,
        class_path=opt.class_path,
        input_shape=image_shape,
        sequence_length=opt.sequence_length,
        training=True,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)


    # Classification criterion
    cls_criterion = nn.CrossEntropyLoss().to(device)

    # Define network
    model = ConvLSTM(
        num_classes=train_dataset.num_classes,
        latent_dim=opt.latent_dim,
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
        attention=True,
    )
    model = model.to(device)

    # Add weights from checkpoint model if specified
    if opt.checkpoint_model:
        model.load_state_dict(torch.load(opt.checkpoint_model))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    def test_model(epoch):
        """ Evaluate the model on the test set """
        print("")
        model.eval()
        test_metrics = {"loss": [], "acc": []}
        for batch_i, (Xf, Xp, y) in enumerate(test_dataloader):
            image_sequences = Variable(Xf.to(device), requires_grad=False)
            pose_sequences = Variable(Xp.to(device), requires_grad=False)
            labels = Variable(y, requires_grad=False).to(device)
            with torch.no_grad():
                # Reset LSTM hidden state
                model.lstm.reset_hidden_state()
                # Get sequence predictions
                predictions = model(image_sequences, pose_sequences)
            # Compute metrics
            acc = 100 * (predictions.detach().argmax(1) == labels).cpu().numpy().mean()
            loss = cls_criterion(predictions, labels).item()
            
            # Keep track of loss and accuracy
            test_metrics["loss"].append(loss)
            test_metrics["acc"].append(acc)
            # Log test performance
            if batch_i==(len(test_dataloader)-1):
                sys.stdout.write(
                    "\rTesting -- [Batch %d/%d] [Loss: %f, Acc: %.2f%% ]"
                    % (
                        batch_i,
                        len(test_dataloader),
                        np.mean(test_metrics["loss"]),
                        np.mean(test_metrics["acc"]),
                    )
                )
        model.train()
        
        print("")
   
    for epoch in range(opt.num_epochs):
        epoch_metrics = {"loss": [], "acc": []}
        prev_time = time.time()
        print('\n')
        print(f"--- Epoch {epoch} ---")
        for batch_i, (Xf, y) in enumerate(train_dataloader):

            if Xf.size(0) == 1:
                continue

            image_sequences = Variable(Xf.to(device), requires_grad=True)
            labels = Variable(y.to(device), requires_grad=False)

            optimizer.zero_grad()

            # Reset LSTM hidden state
            model.lstm.reset_hidden_state()

            # Get sequence predictions
            predictions = model(image_sequences)

            # Compute metrics
            loss = cls_criterion(predictions, labels)
            acc = 100 * (predictions.detach().argmax(1) == labels).cpu().numpy().mean()

            loss.backward(retain_graph=True)
            optimizer.step()

            # Keep track of epoch metrics
            epoch_metrics["loss"].append(loss.item())
            epoch_metrics["acc"].append(acc)

            # Determine approximate time left
            batches_done = epoch * len(train_dataloader) + batch_i
            batches_left = opt.num_epochs * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            if batch_i==(len(train_dataloader)-1):
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f), Acc: %.2f%% (%.2f%%)] ETA: %s"
                    % (
                        epoch,
                        opt.num_epochs,
                        batch_i,
                        len(train_dataloader),
                        loss.item(),
                        np.mean(epoch_metrics["loss"]),
                        acc,
                        np.mean(epoch_metrics["acc"]),
                        time_left,
                    )
                )
            
            # Empty cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Evaluate the model on the test set
        #test_model(epoch)

        # Save model checkpoint
        if epoch % opt.checkpoint_interval == 0:
            os.makedirs("model_checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"model_checkpoints/{model.__class__.__name__}_{epoch}.pth")
