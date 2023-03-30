"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)
            
    
class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)
            
        
class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    sparse: boolean, optional
        Use sparse gradients.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(self, num_users, num_items, embedding_dim=32, layer_sizes=[96, 64], 
                 sparse=False, embedding_sharing=True):

        super().__init__()

        self.embedding_dim = embedding_dim

        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************


        self.U = ScaledEmbedding(num_users, embedding_dim,sparse=sparse)
        self.M = ScaledEmbedding(num_items, embedding_dim,sparse=sparse)

        self.A = ZeroEmbedding(num_users, 1,sparse=sparse)
        self.B = ZeroEmbedding(num_items, 1,sparse=sparse)

        self.mlp=nn.Sequential()
        for i in range(len(layer_sizes) - 1):
            self.mlp.add_module("linear",nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            self.mlp.add_module("relu",nn.ReLU())
        self.mlp.add_module("output",nn.Linear(layer_sizes[-1],1))
        self.embedding_sharing=embedding_sharing
        if embedding_sharing:
            pass
        else:
            self.U1 = ScaledEmbedding(num_users, embedding_dim)
            self.M1 = ScaledEmbedding(num_items, embedding_dim)
            self.A1 = ZeroEmbedding(num_users, 1)
            self.B1 = ZeroEmbedding(num_items, 1)
        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************
        
    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of 
            shape (batch,). This corresponds to p_ij in the 
            assignment.
        score: tensor
            Tensor of user-item score predictions of shape 
            (batch,). This corresponds to r_ij in the 
            assignment.
        """
        
        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************

        users=self.U(user_ids)  #(n,d)
        items= self.M(item_ids) #(n,d)
        # bug code
        # user_items=users*items+self.A(user_ids)+self.B(item_ids)  #(n,d)
        # predictions=(user_items).sum(1)

        user_items=users*items

        predictions=(user_items).sum(1,keepdim=True)+self.A(user_ids)+self.B(item_ids)   #(n,1)
        predictions=torch.squeeze(predictions)

        # task1
        # predictions=(user_items).sum(1)
        # task 2

        if self.embedding_sharing==False:
            users=self.U1(user_ids)  #(n,d)
            items= self.M1(item_ids) #(n,d)
            user_items=users*items  #(n,d)

        x=torch.cat((users,items,user_items),dim=1) #(n,3d)
        score=self.mlp(x)  #(n,1)
        score=torch.squeeze(score)

        #********************************************************
        #********************************************************
        #********************************************************
        return predictions, score
