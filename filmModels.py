import torch

class mlp(torch.nn.Module):
        def __init__(self, input_size, output_size = 1, hidden_width = 20, hidden_nblocks = 2):
            super(mlp, self).__init__()
            self.input_size = input_size
            self.output_size = output_size
            
            self.hidden_width = hidden_width
            self.hidden_nblocks = hidden_nblocks
            
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_width)
            self.fc2 = torch.nn.Linear(self.hidden_width,self.hidden_width)
            self.fc3 = torch.nn.Linear(self.hidden_width, self.output_size)
            
            self.relu = torch.nn.ReLU()
            self.end= torch.nn.Softmax(dim = -1) ## sigmoid for multi-label, softmax for multi-class (mutually exclusive)
            
            self.dropout = torch.nn.Dropout(0.25)
            
        def forward(self, x, film_params):
            out = self.fc1(x)
            out = self.relu(out)
            
            
            for i in range(self.hidden_nblocks):
                out = self.fc2(out)
                
                # ------- film layer -----------
                start = i * hidden_width * 2
                mid = start + hidden_width
                end = mid + hidden_width
                
                gamma = film_params[:, start : mid]
                beta = film_params[:, mid : end]
                
#                 print(out.shape)
#                 print(gamma.shape)
#                 print(beta.shape)
                
                out = out * gamma
                out += beta
                # ------- film layer -----------
                # out = self.dropout(out)
                out = self.relu(out)
            
            out = self.fc3(out)
            # out = self.end(out)
            return out