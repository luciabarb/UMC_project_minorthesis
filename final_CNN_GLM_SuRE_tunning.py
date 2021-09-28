import numpy as np
import h5py
from sklearn import preprocessing
import sys
import os
import pandas as pd
import sys
import time
import glob
import pysam
from argparse import ArgumentParser, RawTextHelpFormatter
from torchsummary import summary
from matplotlib import pyplot as plt, colors
import random
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
import optuna
import gensim

import tempfile, logbook
from optuna.trial import TrialState
import torch.optim as optim
import pypairix #imprort large datasets quicker
import joblib

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD


class Beluga(nn.Module):
    #Define model function
    def __init__(self, trial):
        super(Beluga, self).__init__()

        global filter_size, dropout, n_layers, initial_channels

        if dna2vec_bool:
            initial_channels = 100 #if dna2vec embbeding the initial channels are 100
        else:
            initial_channels = 4

        #Define hyperparameters search
        filter_size = trial.suggest_categorical("filter_size", [32,64,128])
        dropout = trial.suggest_categorical("dropout", [0.1,0.2,0.3,0.6])
        print('Filter size:', filter_size)
        print('Drop out:', dropout)
        n_layers = trial.suggest_int("n_layers", 1, 3)
        kernel_size, kernel_size_2 = 1, 1
        print('Nº layers:', n_layers)

        if n_layers == 1:
            kernel_size = trial.suggest_categorical("kernel_size", [6,10,20])
            print('Kernel size:', kernel_size)
            maxpool  = int(kernel_size/2)
            stride = maxpool
        if n_layers == 2:
            kernel_size_2 = trial.suggest_categorical("kernel_size_2", [6,10])
            maxpool  = int(kernel_size_2/2)
            stride = int(maxpool/2)
            print('Kernel size:', kernel_size_2)



        self.model1layer = nn.Sequential(
                        nn.Conv1d(in_channels=initial_channels,out_channels=filter_size,kernel_size=kernel_size),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=maxpool, stride=stride),
                        nn.BatchNorm1d(filter_size),
                        nn.Dropout(dropout))

        self.modelinit_2 = nn.Sequential(
                        nn.Conv1d(in_channels=initial_channels,out_channels=filter_size,kernel_size=kernel_size_2),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=maxpool, stride=stride),
                        nn.BatchNorm1d(filter_size),
                        nn.Dropout(dropout))

        self.model_2 = nn.Sequential(
                        nn.Conv1d(in_channels=filter_size,out_channels=filter_size,kernel_size=kernel_size_2),
                        nn.ReLU(),
                        nn.MaxPool1d(maxpool,stride=stride),
                        nn.BatchNorm1d(filter_size),
                        nn.Dropout(dropout))
        if n_layers == 1:
            self.model_layers = self.model1layer

        elif n_layers == 2:
            self.model_layers = nn.Sequential(self.modelinit_2, self.model_2)

        elif n_layers == 3:
            self.model_layers = nn.Sequential(self.modelinit_2, self.model_2, self.model_2)

        else:
            print('Something is going wrong---')
            print(n_layers)



        self.out1 = nn.LazyLinear(out_features=1) #Dense layer


    def forward(self, x):
        x = self.model_layers(x) #Output shape: batch_size,channel,seq
        x = x.view(x.size(0), -1) #Flatten
        x = self.out1(x)

        return x

#dna2vec functions, adapted from: https://github.com/pnpnpn/dna2vec/
class SingleKModel:
    def __init__(self, model):
        self.model = model
        self.vocab_lst = sorted(model.vocab.keys())

class MultiKModel:
    def __init__(self, filepath):
        self.aggregate = gensim.models.KeyedVectors.load_word2vec_format(filepath, binary=False)
        self.logger = logbook.Logger(self.__class__.__name__)
        vocab_lens = [len(vocab) for vocab in self.aggregate.vocab.keys()]
        self.k_low = min(vocab_lens)
        self.k_high = max(vocab_lens)
        self.vec_dim = self.aggregate.vector_size
        self.data = {}
        for k in range(self.k_low, self.k_high + 1):
            self.data[k] = self.separate_out_model(k)

    def model(self, k_len):
        """
        Use vector('ACGTA') when possible
        """
        return self.data[k_len].model

    def vector(self, vocab):
        return self.data[len(vocab)].model[vocab]

    def unitvec(self, vec):
        return matutils.unitvec(vec)

    def cosine_distance(self, vocab1, vocab2):
        return np.dot(self.unitvec(self.vector(vocab1)), self.unitvec(self.vector(vocab2)))

    def l2_norm(self, vocab):
        return np.linalg.norm(self.vector(vocab))

    def separate_out_model(self, k_len):
        vocabs = [vocab for vocab in self.aggregate.vocab.keys() if len(vocab) == k_len]
        if len(vocabs) != 4 ** k_len:
            self.logger.warn('Missing {}-mers: {} / {}'.format(k_len, len(vocabs), 4 ** k_len))
        header_str = '{} {}'.format(len(vocabs), self.vec_dim)
        with tempfile.NamedTemporaryFile(mode='w') as fptr:
            print(header_str, file=fptr)
            for vocab in vocabs:
                vec_str = ' '.join("%f" % val for val in self.aggregate[vocab])
                print('{} {}'.format(vocab, vec_str), file=fptr)
            fptr.flush()
            return SingleKModel(gensim.models.KeyedVectors.load_word2vec_format(fptr.name, binary=False))


def subset_dataset(SuRE_count, chr_size, dataset):
    ######################################################
    #Divides dataset into training, validation and test
    #Input:
    ### SuRE_count = pypairix object to loop through the data quicker
    ### chr_size = dictionary with chromsomes sizes 'chrX':...
    ### dataset = specify if dataset is test, validation or training set
    #######################################################


    #When splitting validation, training and test set by chromsomes
    if 'TSS' not in output_directory:
        chr_test = ['chr10']
        chr_val = ['chr8', 'chr17']

        if dataset == 'test':
            chr_list = chr_test
        elif dataset == 'validation':
            chr_list = chr_val
        elif dataset == 'train':
            chr_list = list(set(n_chr)-set(chr_test)-set(chr_val))


        list_chr = []
        for chr in chr_list:
            it = SuRE_count.query(chr, 0 , int(chr_size[chr]))
            list_chr.extend(list(it))
        SuRE_count_final = pd.DataFrame(list_chr, columns=['chr', 'start', 'end', 'strand','score'])
        if dna2vec_bool:
            SuRE_count_final = SuRE_count_final.sample(n= int(SuRE_count_final.shape[0]*0.3), random_state = 47)

    else: #When splitting validation, training and test set by random subset of TSS
        if dataset == 'test':
            SuRE_count = pypairix.open(''.join([SuRE_count,'_test.txt.gz'])) #Files already splitted, otherwise it took a long time to run every time
        elif dataset == 'validation':
            SuRE_count = pypairix.open(''.join([SuRE_count,'_validation.txt.gz']))
        elif dataset == 'train':
            SuRE_count = pypairix.open(''.join([SuRE_count,'_train.txt.gz']))


        list_chr = []
        for chr in n_chr:
            it = SuRE_count.query(chr, 0 , chr_size[chr])
            list_chr.extend(list(it))

    SuRE_count_final = pd.DataFrame(list_chr, columns=['chr', 'start', 'end', 'strand','score'])

    if dna2vec_bool: #If dna2vec is used, subsample to 30% of samples, as the dataset is too big otherwise
        SuRE_count_final = SuRE_count_final.sample(n= int(SuRE_count_final.shape[0]*0.3), random_state = 47)


        os.path.join(main_directory,file)


    print('----------- Only ' + str(SuRE_count_final.shape[0]) + ' sequences selected', flush=True)

    return(SuRE_count_final)


def format_data_as_tensors(SuRE_count_norm):
    ######################################################
    #Retrieve sequence and transform into one-hot encoding
    #Input:
    ### SuRE_count = Normalized dataframe of SuRE counts
    ###              [0]: chr, [1]: start, [2]: end,
    ###                  [3]: strand, [4]: SuRE scores
    #######################################################

    Y = SuRE_count_norm.iloc[:,4:].values

    def retrieve_sequence(df_coordinates):
        ######################################################
        #Gives the sequence given chr, start and end position
        #Input:
        ### df_coordinates = [0]: chr, [1]: start, [2]: end,
        ###                  [3]: strand
        ###
        #######################################################
        my_comp = ''.maketrans('ACGT','TGCA')
        sequence = hg19.fetch(df_coordinates[0],int(df_coordinates[1])-1,int(df_coordinates[2]))
        #It has to use -1 in the start cause it has to include df_coordinates[2] and df_coordinates[1] in the sequence.
        #The -1 is used because otherwise the first nt will be missing --> check http://genome.ucsc.edu/cgi-bin/das/hg19/dna?segment=chr1:51047076,51047085
        #I have double check it: e.g. for the new data there are some SNPs in "SNPrelpos" (=relative position) 0, if we don't use the -1 it will not include these SNPs

        sequence = sequence.upper() #There are regions in the genome that have lower case
        #If it's negative strand reverse it
        if df_coordinates[3] == '-':
            sequence = sequence.translate(my_comp)[::-1]
        return(sequence)

    df_coordinates = SuRE_count_norm.iloc[:,:4].values
    seq = [retrieve_sequence(df_coordinates[i,:]) for i in range(df_coordinates.shape[0])]

    #dna2vec
    def Gen_Words(sequences,kmer_len,stride):
        ######################################################
        #Transforms sequences to kmers and then vectors with dna2vec
        #Input:
        ### sequences = list of sequences
        ### kmer_len = length of k-kmers
        ### stride
        #######################################################
        filepath_model = os.path.join(main_directory,'dna2vec','pretrained','dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v')
        mk_model = MultiKModel(filepath_model)
        out=[]
        for seq in sequences:
            kmer_list=[]
            for j in range(0,(len(seq)-kmer_len)+1,stride):
                kmer = seq[j:j+kmer_len]
                vector = np.array(mk_model.vector(kmer)) # 100 channels

                kmer_list.append(vector)  # This will have shape (seq, channel)


            out.append(np.array(kmer_list))

        out = np.array(out)# This will have shape (batch, seq, channel)

        return out

    #One hot encoding
    def one_hot_encode(sequences):
        ######################################################
        #Gives the sequence encoded in one hot coding with padding
        ## in the sides
        #Input:
        ### sequences = sequences list
        #######################################################

        ACTG_numeric = {'A':np.array([1.,0.,0.,0.]),'C':np.array([0.,1.,0.,0.]),'G':np.array([0.,0.,1.,0.]),'T':np.array([0.,0.,0.,1.]),'N':np.array([0.,0.,0.,0.])};

        X_OneHot = []
        for seq in sequences:
            x = np.array([ACTG_numeric[s] for s in seq])

            X_OneHot.append(x)

        X_OneHot = np.array(X_OneHot)

        return(X_OneHot)

    if dna2vec_bool:

        idx_noNs = [n for n,s in enumerate(seq) if "N" not in s] #Remove sequences with N, otherwise you'll get an error (dna2vec dict. doesn't contain Ns)

        SuRE_count_norm = SuRE_count_norm.iloc[idx_noNs]
        seq = [seq[i] for i in idx_noNs]

        Y = SuRE_count_norm.iloc[:,4:].values

        X = Gen_Words(seq, 3, 1)

    else:
        X = one_hot_encode(seq)
        print(X, flush=True)

    X = torch.from_numpy(X)
    Y =  torch.from_numpy(Y)

    return(X, Y)


def obtain_split_datasets(bin_size):
    #######Load reference genome hg19######
    global hg19
    hg19_directory = os.path.join('hg19_genome','hg19.fa')
    hg19 = pysam.FastaFile(os.path.join(main_directory,hg19_directory))

    ######Get size of chr######
    global n_chr, chr_size
    n_chr = list(range(1,23)); n_chr.append('X')
    n_chr = list(map(lambda x: 'chr' + str(x), n_chr))

    df_chrsize = pd.read_csv(os.path.join(main_directory, 'hg19_genome','size_hg19_chr.txt'), names = ['chr','size_chr'], sep = '\t')

    df_chrsize = df_chrsize[df_chrsize['chr'].isin(n_chr)]
    chr_size = dict(zip(df_chrsize.chr, df_chrsize.size_chr))

    ###### Open file #####

    gz_file = os.path.join(main_directory,file)
    df_SuRE_info = pypairix.open(os.path.join(main_directory,gz_file))


    ########OBTAIN TEST SET######
    print('--LOADING TEST DATASET', flush=True)
    SuRE_count_norm_test = subset_dataset(df_SuRE_info, chr_size, 'test')
    x_test, y_test = format_data_as_tensors(SuRE_count_norm_test)

    ########OBTAIN VALIDATION SET######
    print('--LOADING VALIDATION DATASET', flush=True)
    SuRE_count_norm_val = subset_dataset(df_SuRE_info, chr_size, 'validation') #SuRE_count_norm_test) CHANGE
    x_val, y_val = format_data_as_tensors(SuRE_count_norm_val)

    ########TRAINING SET

    print('--LOADING TRAINING DATASET', flush=True)
    SuRE_count_norm_train = subset_dataset(df_SuRE_info, chr_size, 'train') #change, should be SuRE_count_val_test
    x_train, y_train = format_data_as_tensors(SuRE_count_norm_train)

    return(x_train, y_train, x_val, y_val, x_test, y_test)


def parse_args():
    "Parses inputs from commandline and returns them as a Namespace object."

    parser = ArgumentParser(prog = 'python3 test.py',
        formatter_class = RawTextHelpFormatter, description =
        '  .\n\n'
        '  Example syntax:\n'
        '     srun --time 30000 -J CNN_GLM --mem 150 -p gpu --gpus-per-node=2 -v --mail-type=FAIL --mail-user=barbadillalucia@gmail.com --error=../output/error_dna2vec_CNN_hyperparameters.txt ../../anaconda3/bin/python3.9 CNN_GLM_SuRE_tunning.py ../raw/ SuRE_elasticnet/SURE_elasticNet_allele_K562_plus_promoter_windows200bin_sum.gz ../output/SuRE_elasticnet/ 200 1'
        '     srun --time 30000 -J CNN_GLM --mem 150 -p gpu --gpus-per-node=2 -v --mail-type=FAIL --mail-user=barbadillalucia@gmail.com --error=../output/error_dna2vec_CNN_hyperparameters.txt ../../anaconda3/bin/python3.9 CNN_GLM_SuRE_tunning.py ../raw/ SuRE_elasticnet/SURE_elasticNet_allele_K562_plus_promoter_windows200bin_sum ../output/SuRE_elasticnet/ 200 1') #If validation,test,train altready splitted

    def str2bool(v):
        if isinstance(v, bool):
            return vHT1080
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # Arguments for the input files
    parser.add_argument('dir_input', help='directory with input files (raw table)')

    parser.add_argument('file', help='.gz file name')

    parser.add_argument('out_dir',  help='path to a directory where output files are saved')

    parser.add_argument('bin_size', help='Bin size (bp)', type=int)

    parser.add_argument('dna2vec', help='0==False, 1==True', type=int, default=0)


    return parser.parse_args()


def objective(trial):
    ##Necesary function for OPTU
    model = Beluga(trial)
    print('Model', model)


    lr = trial.suggest_float('lr', 1e-4, 1e-2)

    # defining the optimizer
    optimizer = Adam(model.parameters(), lr=lr)

    # defining the loss function
    criterion = nn.MSELoss()

    batch_size = trial.suggest_categorical('batch_size', [32, 64,256,1024])

    print('Batch size:', batch_size)
    n_epochs = 25
    print('Nº epoch:', n_epochs)
    print('Learning rate: ', lr )

    train_loader = DataLoader(dataset_train, batch_size= batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size= batch_size, shuffle=True)

    # empty list to store training and validation losses
    train_losses, val_losses = [], []
    results = []

    for epoch in range(n_epochs):
        print('-------------------------------------------')
        print('--- Epoch ', epoch)

        model.train()

        current_loss,training_loss = 0.0, 0.0
        y_train_predicted, y_train_true, training_loss = train_loop(train_loader, model, criterion, optimizer)


        model.eval()
        with torch.no_grad():
            y_val_predicted, y_val_true, val_loss = validation_loop(val_loader, model, criterion)

            results.append([epoch, training_loss, val_loss])

        trial.report(val_loss, epoch)

        #Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    plt.hist2d(y_val_true, y_val_predicted, bins=(50, 50), norm=colors.LogNorm())
    plt.xlabel('y valid')
    plt.ylabel('y val predicted')
    plt.colorbar()
    plt.savefig(os.path.join(output_directory,"y_validation_pred_vs_real_batch"+str(batch_size)+'_lr'+str(lr)+'_filter'+str(filter_size)+'_dropout'+str(dropout)+'_nlayers'+str(n_layers)+'.png'), bbox_inches='tight')
    plt.clf()

    # Process is complete.


    return(val_loss)

def train_loop(dataloader, model, criterion, optimizer):
    # checking if GPU is available
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    training_loss = 0.0
    y_train_predicted,  y_train_true = np.array([]), np.array([])

    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()

        pred = model(X)
        loss = criterion(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

        if batch % 5 == 0:
            y_train_predicted = np.append(y_train_predicted, pred.cpu().detach().numpy().flatten())
            y_train_true = np.append(y_train_true, y.cpu().detach().numpy().flatten())

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    training_loss /= batch

    return(y_train_predicted, y_train_true, training_loss)

def validation_loop(dataloader, model, criterion):
    size = len(dataloader.dataset)
    val_loss = 0

    y_val_predicted,  y_val_real = np.array([]), np.array([])

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()

            pred = model(X)
            val_loss += criterion(pred, y).item()

            y_val_predicted = np.append(y_val_predicted, pred.cpu().detach().numpy().flatten())
            y_val_real = np.append(y_val_real, y.cpu().detach().numpy().flatten())

    val_loss /= batch
    print(f"Test Error: Avg loss: {val_loss:>8f} \n")

    return(y_val_predicted, y_val_real, val_loss)


def main(args=False):
    "Perform CNN for SuRE data."
    if not args: args = parse_args()

    #ARGUMENTS

    global main_directory, file
    main_directory = args.dir_input
    file = args.file
    bin_size = args.bin_size
    print('File name:', file)

    global dna2vec_bool
    dna2vec_bool = True if args.dna2vec == 1 else False

    global output_directory
    cell_line = 'HEPG2' if 'HEPG2' in file else 'K562'
    strand = 'plus' if 'plus' in file else 'minus'

    output_directory = os.path.join(args.out_dir,'CNN_elasticnet_'+cell_line+'_'+strand,str(bin_size),'hyperparameter_tunning')
    if dna2vec_bool:
        output_directory = os.path.join(output_directory,'dna2vec')

    print(output_directory)
    if os.path.exists(output_directory): #Create trial directories if folder already exists
        for i in range(2,100):
            trial_directory = os.path.join(output_directory,'trial_'+str(i))
            if not os.path.exists(trial_directory):
                break
        output_directory = trial_directory

    if not os.path.exists(output_directory): os.makedirs(output_directory) #Create folder where all the output is going to be saved


    #All printing functions will be saved in a file
    f = open(os.path.join(output_directory, 'screen_messages_'+cell_line+'_'+strand+'.txt'), 'w')
    sys.stdout = f


    ##########Obtain dataset##########
    x_train, y_train, x_val, y_val = obtain_split_datasets(bin_size)
    x_train, x_val = x_train.permute(0,2,1), x_val.permute(0,2,1) #permute tensors to have the right size, (n_batch,channel,seq)
    global dataset_train, dataset_val
    dataset_train , dataset_val = TensorDataset(x_train, y_train), TensorDataset(x_val, y_val)


    ##########MODEL TRAINING AND HYPERPARAMETER TUNNING WITH OPTUNA##########
    print('Start study', flush='True')
    study = optuna.create_study(study_name='CNN_olddataset',direction='minimize') #bayesian optimization
    study.optimize(objective, n_trials=50)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])


    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


    joblib.dump(study, os.path.join(output_directory,"study.pkl"))

    importance_hyperparam = optuna.importance.get_param_importances(study)


    print(importance_hyperparam)

    f.close()



if __name__ == '__main__':
	main()
