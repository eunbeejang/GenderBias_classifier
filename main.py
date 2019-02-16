import torch
from torch.autograd import Variable

import data
import model
import train
#from tqdm import tqdm # progress bar


# not pretty ask the mentors for a better hyperparm tuning methods
def hyperpram_tune(train_data, valid_data, test_data, optimizer, seq_len, emb_dim, output_dim, word_emb, batch_size, max_vocab_size, hidden_dim_lst = [64,128,256], lr_lst = [0.1,0.01,0.001], max_grad_norm_lst = [3,4,5], epoch_lst = [10,20,30,40,50]):
    best_valid_loss = 0
    best_model = ""
    for dim in hidden_dim_lst:
        for rate in lr_lst:
            for norm in max_grad_norm_lst:
                for epoch in epoch_lst:
                    lr = rate
                    max_grad_norm = norm
                    print("&&&& hidden_dim {}, lr {}, max_grad_norm {}, epoch {}".format(dim,rate,norm,epoch))
                    this_model = model.LSTM(seq_len, emb_dim, dim, output_dim, word_emb, batch_size, max_vocab_size)
                    print(this_model)
                    _, _ = train.model_train(this_model, train_data, valid_data, test_data, epoch , optimizer, max_grad_norm).train()  
                    avg_valid_loss, _, _ = train.model_train(this_model, train_data, valid_data, test_data, epoch , optimizer, max_grad_norm).evaluate()
                    if avg_valid_loss > best_valid_loss:
                        best_model = this_model
    return best_model


def main():

	batch_size = 64
	max_vocab_size = 25000
	seq_len = 200
	emb_dim = 300
	hidden_dim = 256
	output_dim = 2
	lr = 0.001
	max_grad_norm = 5
	epoch_size = 20

	# load data
	data_ld = data.dataloader(batch_size, max_vocab_size)
	TEXT, word_emb, train_data, valid_data, test_data, vocab_size = data_ld.load_data()

	lstm = model.LSTM(seq_len, emb_dim, hidden_dim, output_dim, word_emb, batch_size, max_vocab_size)
	optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, lstm.parameters()), lr=lr )
	

	# save the best model
	best_model = hyperpram_tune(train_data, valid_data, test_data, optimizer, seq_len, emb_dim, output_dim, word_emb, batch_size, max_vocab_size)
	print("The best model is: \n")
	print(best_model)
#	with open(args.save, 'wb') as f:
#		torch.save(model, f)
	torch.save(best_model.state_dict(), './best_model.pt')



if __name__ == '__main__':
	main()