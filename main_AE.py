import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from Extended_data_visual import N_E, N_CV, N_T, T
from visual_supplementary import create_dataset, visualize_similarity, y_size
from Extended_data_visual import DataLoader_GPU

def train(model, H_data_train,H_data_valid, num_epochs, batch_size, learning_rate):
    torch.manual_seed(42)
    #criterion_x = nn.MSELoss()
    criterion_y = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    train_loader = torch.utils.data.DataLoader(H_data_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(H_data_valid, batch_size=len(H_data_valid), shuffle=True)
    outputs = []
    epoch_list = []
    Loss_train_list = []
    Loss_valid_list = []
    for epoch in range(num_epochs):
        for data in train_loader:
            state_batch, img_batch = data
            x_bottom_batch,recon_batch = model(img_batch)

            #check_learning_process(img_batch, recon_batch, epoch, 'Train')

            loss_y = criterion_y(recon_batch, img_batch)
            #loss_x = criterion_x(x_bottom_batch, state_batch)
            #Total_loss = loss_y + loss_x
            loss_y.backward()
            optimizer.step()
            optimizer.zero_grad()

            #validation
            for data in val_loader:
                state_batch, img_batch = data
                x_bottom_batch, recon_batch = model(img_batch)
                #check_learning_process(img_batch, recon_batch, epoch, 'Val')
                loss_y_valid = criterion_y(recon_batch, img_batch)

        epoch_list.append(epoch)
        Loss_valid_list.append(float(loss_y_valid))
        Loss_train_list.append(float(loss_y))
        print('Epoch:{}, Train Loss:{:.7f}, Valid Loss:{:.7f}'.format(epoch+1, float(loss_y), float(loss_y_valid)))
    outputs.append((state_batch, x_bottom_batch, img_batch, recon_batch))
    return outputs, epoch_list, Loss_train_list, Loss_valid_list, model

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            torch.nn.Linear(y_size * y_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 5)
        )
        self.decoder = nn.Sequential(
            torch.nn.Linear(5, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, y_size * y_size),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x_bottom = x
        y_rec = self.decoder(x_bottom)
        return x_bottom,y_rec

def check_learning_process(img_batch,recon_batch,epoch, name):
    y_nump = img_batch[1].reshape(y_size,y_size).detach().numpy().squeeze()
    y_recon_nump = recon_batch[1].reshape(y_size,y_size).detach().numpy().squeeze()
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 2, 1)
    plt.imshow(y_nump)
    plt.axis('off')
    plt.title("origin")

    fig.add_subplot(1, 2, 2)
    plt.imshow(y_recon_nump)
    plt.axis('off')
    plt.title("reconstruct")
    fig.savefig('AE Process/{} Process at epoch {}'.format(name,epoch))

def train_AE():
    dataFolderName = 'Simulations/Synthetic_visual' + '/'
    dataFileName = 'y{}x{}_T{}_NE{}_NT{}_NCV{}_Sigmoid.pt'.format(y_size,y_size, T,N_E,N_T,N_CV)
    [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader_GPU(dataFolderName + dataFileName)
    H_data_train = create_dataset(train_input, train_target, N_E)
    H_data_valid = create_dataset(cv_input, cv_target, N_CV)

    model = Autoencoder()
    max_epochs = 20
    BATCH_ZISE = 128
    LR = 1e-3
    outputs, epoch_list, Loss_train_list, Loss_valid_list, AE_model = train(model, H_data_train,H_data_valid, num_epochs=max_epochs, batch_size=BATCH_ZISE, learning_rate=LR)
    return AE_model

    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 1, 1)
    plt.plot(epoch_list, Loss_train_list, label='Train')
    plt.plot(epoch_list, Loss_valid_list, label='Val')
    plt.title("AutoEncoader Loss")
    fig.savefig('AE Process/Loss')

    visualize_similarity(outputs[0][2][22].squeeze().detach().numpy().reshape((y_size,y_size)), outputs[0][3][22].squeeze().detach().numpy().reshape((y_size,y_size)))

if __name__ == '__main__':
    train_AE()


