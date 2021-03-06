import torch
from torch.nn import functional

import copy
import model
"""
def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
        
"""
class model_train(object):

    num_of_train = 0
    num_of_eval = 0

    def __init__(self, model, train_data, valid_data, test_data, epoch_size, optimizer, max_grad_norm):
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.epoch_size = epoch_size
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm


    def train(self):

        model_train.num_of_train += 1

        self.model.train()

        total_loss = []
        total_acc = []
        
        #for epoch in tqdm(range(self.epoch_size)):
        for epoch in range(self.epoch_size):
            epoch_loss = []
            epoch_acc = []
            
            for i, batch in enumerate(self.train_data):
                self.optimizer.zero_grad() 

                batch_size = len(batch.text[0])
                pred = self.model(batch.text[0],batch_size)
                loss = functional.cross_entropy(pred, batch.label, size_average=False)            
                correct = ((torch.max(pred, 1)[1] == batch.label)).sum().numpy()
                acc = correct/pred.shape[0]
                
                epoch_loss.append(loss.item())
                epoch_acc.append(acc)
         
                loss.backward() # calculate the gradient
            
                # Clip to the gradient to avoid exploding gradient.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                #clip_gradient(model, 0.25) # limit the norm

                self.optimizer.step() # update param

                print("------TRAINBatch {}/{}, Batch Loss: {:.4f}, Accuracy: {:.4f}".format(i+1,len(self.train_data), loss, acc))
            
            total_loss.append((sum(epoch_loss)/len(self.train_data)))
            total_acc.append((sum(total_acc)/len(self.train_data)))
            print("****** Epoch {} Loss: {}, Epoch {} Acc: {}".format(epoch, (sum(epoch_loss)/len(self.train_data)),
                                                                      epoch, (sum(epoch_acc)/len(self.train_data))))          
        return total_loss, total_acc

    def evaluate(self):

        model_train.num_of_eval += 1

        self.model.eval()

        total_loss = []
        total_acc = []

        for i, batch in enumerate(self.valid_data):
            batch_size = len(batch.text[0])
            pred = model(batch.text[0],batch_size)
            loss = functional.cross_entropy(pred, batch.label, size_average=False)            
            correct = ((torch.max(pred, 1)[1] == batch.label)).sum().numpy()
            acc = correct/pred.shape[0]
            total_loss.append(loss.item())
            total_acc.append(acc)
            print("++++++EVAL Batch {}/{}, Batch Loss: {:.4f}, Accuracy: {:.4f}".format(i+1,len(self.valid_data), loss, acc))
        print("Average EVAL Loss: ", (sum(total_loss) / len(self.valid_data))) 
        print("Average EVAL Acc: ", (sum(total_acc) / len(self.valid_data))) 
        return avg_total_loss, total_loss, total_acc


