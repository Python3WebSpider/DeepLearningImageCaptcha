# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import dataset
from model import CNN
from test import main as test

# Hyper Parameters
num_epochs = 30
batch_size = 100
learning_rate = 0.001



def main():
    cnn = CNN()
    cnn.train()
    print('init net')
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    
    max_test_acc = -1
    # Train the Model
    train_dataloader = dataset.get_train_data_loader()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            images = Variable(images)
            labels = Variable(labels.float())
            predict_labels = cnn(images)
            # print(predict_labels.type)
            # print(labels.type)
            loss = criterion(predict_labels, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print("epoch:", epoch, "step:", i, "loss:", loss.item())
            if (i + 1) % 100 == 0:
                torch.save(cnn.state_dict(), "./model.pkl")  # current is model.pkl
                print("save model")
        print("epoch:", epoch, "step:", i, "loss:", loss.item())
        test_acc = test()
        if test_acc > max_test_acc:
            torch.save(cnn.state_dict(), "./best_model.pkl")  # best model save as best_model.pkl
            print("save best model")
    torch.save(cnn.state_dict(), "./model.pkl")  # current is model.pkl
    print("save last model")


if __name__ == '__main__':
    main()
