import torch
from torch.autograd import Variable

import data, model, train
#from tqdm import tqdm # progress bar

def main():


	TEXT, word_emb, train_data, valid_data, test_data, vocab_size = data.load_data()

	seq_len = 200
	emb_dim = 300
	hidden_dim = 256
	output_dim = 2
	lr = 0.001
	max_grad_norm = 5
	optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=lr )
	#model = model.LSTM(seq_len, emb_dim, hidden_dim, output_dim, word_emb, batch_size)
	

	def hyperpram_tune(hidden_dim_lst = [64,128,256], lr_lst = [0.1,0.01,0.001], max_grad_norm_lst = [3,4,5], epoch_lst = [10,20,30,40,50]):
	    best_valid_loss = 0
	    best_model = self.model
	    for dim in hidden_dim_lst:
	        for rate in lr_lst:
	            for norm in max_grad_norm_lst:
	                for epoch in epoch_lst:
	                    lr = rate
	                    max_grad_norm = norm
	                    print("&&&& hidden_dim {}, lr {}, max_grad_norm {}, epoch {}".format(dim,rate,norm,epoch))
	                    this_model = model.LSTM(seq_len, emb_dim, dim, output_dim, word_emb, batch_size)
	                    print(this_model)
	                    _, _ = train.model_train(this_model, self.train_data, epoch , optimizer).train()  
	                    avg_valid_loss, _, _ = train.model_train(this_model, self.train_data, epoch , optimizer).evaluate()
	                    if avg_valid_loss > best_valid_loss:
	                        best_model = this_model


	    return best_model

	# save the best model
	best_model = hyperpram_tune()
	print(best_model)
	with open(args.save, 'wb') as f:
		torch.save(model, f)

if __name__ == '__main__':
	main()