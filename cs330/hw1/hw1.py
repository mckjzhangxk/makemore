import argparse
import os
import torch

import torch.nn.functional as F

from torch import nn
from load_data import DataGenerator
from dnc import DNC
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch.utils.tensorboard import SummaryWriter


class MANN(nn.Module):

    def __init__(self, num_classes, samples_per_class, model_size=128, input_size=784):
        super(MANN, self).__init__()
        
        def initialize_weights(model):
            nn.init.xavier_uniform_(model.weight_ih_l0) #这是4个gate的W
            nn.init.xavier_uniform_(model.weight_hh_l0) #这是4个gate的W
            nn.init.zeros_(model.bias_hh_l0)  #这应该是初始cell输入？
            nn.init.zeros_(model.bias_ih_l0)  #这是4个gate的bias
    
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.input_size = input_size
        self.hidden_size=model_size
        self.layer1 = torch.nn.LSTM(num_classes + input_size, 
                                    model_size, 
                                    batch_first=True)
        self.layer2 = torch.nn.LSTM(model_size,
                                    num_classes,
                                    batch_first=True)
        initialize_weights(self.layer1)
        initialize_weights(self.layer2)
        
        # self.dnc = DNC(
        #                input_size=num_classes + input_size,
        #                output_size=num_classes,
        #                hidden_size=model_size,
        #                rnn_type='lstm',
        #                num_layers=1,
        #                num_hidden_layers=1,
        #                nr_cells=num_classes,
        #                cell_size=64,
        #                read_heads=1,
        #                batch_first=True,
        #                gpu_id=0,
        #                )
    def forward_fix(self, input_images, input_labels):
        K=self.samples_per_class
        B=input_labels.shape[0]
        N=self.num_classes
        
        Dtrain=torch.cat((input_images[:,0:K,:,:],input_labels[:,0:K,:,:]),dim=3)
        Dtest=torch.cat((input_images[:,K:K+1,:,:],torch.zeros_like(input_labels[:,K:K+1,:,:])),dim=3)  
        
        # 修正,独立predict每个 test
        xtrain=Dtrain.view(B,K*N,Dtrain.shape[-1])
        y1,state_1=self.layer1(xtrain)
        assert tuple(y1.shape)==(B,K*N,self.hidden_size)
        y2,state_2=self.layer2(y1)
        assert tuple(y2.shape)==(B,K*N,N)
        
        xtext=Dtest.view(B,1*N,Dtest.shape[-1])
        
        logits=[]
        for n in range(N):
            y_test_1,_=self.layer1(xtext[:,n:n+1,:],state_1)
            y_test_2,_=self.layer2(y_test_1,state_2)  #(B,1,N)
            assert tuple(y_test_2.shape)==(B,1,N)
            logits.append(y_test_2)
        
        logits=torch.stack(logits,dim=1)    #(B,N,N)
        assert tuple(logits.shape)==(B,N,N)
        y2=torch.cat((y2,logits),dim=1)
        assert tuple(y2.shape)==(B,K*N+N,N)
        
        y2=y2.view(B,K+1,N,N)
        # 修正完成 
        return y2
    def forward(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: tensor
                A tensor of shape [B, K+1, N, 784] of flattened images
            
            labels: tensor:
                A tensor of shape [B, K+1, N, N] of ground truth labels
        Returns:
            
            out: tensor
            A tensor of shape [B, K+1, N, N] of class predictions
        """
        #############################
        #### YOUR CODE GOES HERE ####
        #############################
        K=self.samples_per_class
        B=input_labels.shape[0]
        N=self.num_classes
        
        # 剽窃的答案
        # input_labels_1=input_labels[:,0:K,:,:]
        # input_labels_2=torch.zeros_like(input_labels[:,K:K+1,:,:])
        # input_labels=torch.cat((input_labels_1,input_labels_2),dim=1)
        # x=torch.cat((input_images,input_labels),dim=3)
        # x=x.view(B,-1,x.shape[-1])
        #####

        #我的答案
        Dtrain=torch.cat((input_images[:,0:K,:,:],input_labels[:,0:K,:,:]),dim=3)
        Dtest=torch.cat((input_images[:,K:K+1,:,:],torch.zeros_like(input_labels[:,K:K+1,:,:])),dim=3)  
        x=torch.cat((Dtrain,Dtest),dim=1)
        x=x.view(B,K*N+N,x.shape[-1]) 
        #####
        
 
        
        ####################bug code####################
        # Dtrain=torch.cat([input_images[:,0:K,:,:],input_labels[:,0:K,:,:]],dim=3)
        # Dtest=torch.cat([input_images[:,K:K+1,:,:],torch.zeros_like(input_labels[:,K:K+1,:,:])],dim=3)

        
        # Dtrain= Dtrain.view(B,-1,Dtrain.shape[-1])
        # # bug 写成Dtest = Dtrain.view(B, -1, Dtest.shape[-1])
        # Dtest = Dtest.view(B, -1, Dtest.shape[-1])
        # # 先train 然后test排列，而不是 train,test交替排列,最终导致模型什么都没有学到！！
        # x=torch.cat((Dtrain,Dtest),dim=1)
        ####################
        
        # 
        y1, cells_1=self.layer1(x)


        # h0 = torch.zeros( 1,B, self.num_classes).to(device=input_labels.device)
        # c0 = torch.zeros( 1,B, self.num_classes).to(device=input_labels.device)
        y2,cells_2=self.layer2(y1)

        y2=y2.view(B,K+1,N,N)
        return y2
        # SOLUTION:


    def loss_function(self, preds, labels):
        """
        Computes MANN loss
        Args:
            preds: tensor
                A tensor of shape [B, K+1, N, N] of network outputs
            
            labels: tensor
                A tensor of shape [B, K+1, N, N] of class labels
                
        Returns:
            scalar loss
        """
        #############################
        #### YOUR CODE GOES HERE ####
        #############################
        N=self.num_classes
        logits=preds[:,-1,:,:]
        labels=labels[:,-1,:,:].argmax(dim=-1)
        
        logits=logits.transpose(1,2)

        loss=nn.functional.cross_entropy(logits,labels,reduction='mean')
        return loss
        # SOLUTION:        



def train_step(images, labels, model, optim):
    predictions = model(images, labels)
    loss = model.loss_function(predictions, labels)
    
    optim.zero_grad()
    loss.backward()
    optim.step()
    return predictions.detach(), loss.detach()


def model_eval(images, labels, model):
    predictions = model(images, labels)
    loss = model.loss_function(predictions, labels)
    return predictions.detach(), loss.detach()


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(config.logdir)

    # Download Omniglot Dataset
    if not os.path.isdir('./omniglot_resized'):
        gdd.download_file_from_google_drive(file_id='1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI',
                                            dest_path='./omniglot_resized.zip',
                                            unzip=True)
    assert os.path.isdir('./omniglot_resized')

    # Create Data Generator
    data_generator = DataGenerator(config.num_classes, 
                                   config.num_samples, 
                                   device=device)

    # Create model and optimizer
    model = MANN(config.num_classes, config.num_samples, 
                 model_size=config.model_size)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr = 1e-3)
    
    for step in range(config.training_steps):
        images, labels = data_generator.sample_batch('train', config.meta_batch_size)
        _, train_loss = train_step(images, labels, model, optim)

        if (step + 1) % config.log_every == 0:
            images, labels = data_generator.sample_batch('test', 
                                                         config.meta_batch_size)
            pred, test_loss = model_eval(images, labels, model)
            pred = torch.reshape(pred, [-1, 
                                        config.num_samples + 1, 
                                        config.num_classes, 
                                        config.num_classes])
            pred = torch.argmax(pred[:, -1, :, :], axis=2)
            labels = torch.argmax(labels[:, -1, :, :], axis=2)
            
            trloss,tsloss,acc=train_loss.cpu().numpy(),test_loss.cpu(),pred.eq(labels).double().mean().item()
            print(f"{step}: trainLoss {trloss:.4f},test loss {tsloss:.4f}, acc:{acc:.3f}")
            writer.add_scalar('Train Loss', train_loss.cpu().numpy(), step)
            writer.add_scalar('Test Loss', test_loss.cpu().numpy(), step)
            writer.add_scalar('Meta-Test Accuracy', 
                              pred.eq(labels).double().mean().item(),
                              step)
    torch.save(
        dict(model_parameters=model.state_dict(),               
        optimizer_state_dict=optim.state_dict()),
        f"{config.num_classes}_way-{config.num_samples}_shot_{config.training_steps}.pt"
    )
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--meta_batch_size', type=int, default=128)
    parser.add_argument('--logdir', type=str, 
                        default='run/log')
    parser.add_argument('--training_steps', type=int, default=10000)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--model_size', type=int, default=128)
    main(parser.parse_args())