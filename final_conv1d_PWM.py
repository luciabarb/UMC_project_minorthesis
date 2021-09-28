##Code adapted from DanQ and FactorNet

import torch
import torch.nn as nn
from __future__ import print_function
import numpy as np
import h5py
import scipy.io
np.random.seed(42) # for reproducibility
import pypairix #imprort large datasets quick
import os
import pandas as pd
import pysam
import seqlogo
from matplotlib import pyplot as plt, colors
import seaborn as sns




#Load model
class Beluga(nn.Module):
    def __init__(self):
        super(Beluga, self).__init__()
        self.modeltest = nn.Sequential( nn.Conv1d(in_channels=4,out_channels=128,kernel_size=10),
                            nn.ReLU(),
                            nn.MaxPool1d(3),
                            nn.Dropout(0.2))
        self.out1 = nn.LazyLinear(out_features=1)
    def forward(self, x):
        x = self.modeltest(x)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.out1(x)
        return x


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


def load_test_data(file_testset):
    SuRE_count = pypairix.open(os.path.join(file_testset))

    ######Get size of chr######
    n_chr = list(range(1,23)); n_chr.append('X')
    n_chr = list(map(lambda x: 'chr' + str(x), n_chr))
    df_chrsize = pd.read_csv( os.path.join(main_directory, 'hg19_genome','size_hg19_chr.txt'), names = ['chr','size_chr'], sep = '\t')

    df_chrsize = df_chrsize[df_chrsize['chr'].isin(n_chr)]
    chr_size = dict(zip(df_chrsize.chr, df_chrsize.size_chr))
    chr_test = ['chr10']

    ######Get hg19 file######
    hg19 = pysam.FastaFile(os.path.join(main_directory,'hg19_genome','hg19.fa'))

    ###### Open data ans transform it to dataframe#####
    list_chr = []
    for chr in chr_test:
        it = SuRE_count.query(chr, 0 , int(chr_size[chr]))
        list_chr.extend(list(it))

    x_test = pd.DataFrame(list_chr, columns=['chr', 'start', 'end', 'strand','score'])

    df_coordinates = x_test.iloc[:,:4].values
    seq = [retrieve_sequence(df_coordinates[i,:]) for i in range(df_coordinates.shape[0])]
    X = one_hot_encode(seq)

    x_test = torch.tensor(np.float32(X)).permute(0,2,1)

    return(x_test)


def obtain_PPM(n_filters, kernel_size, x_test):
    #Load 1st CONV1d layer model
    model_CNN = nn.Conv1d(4, n_filters, kernel_size=(kernel_size,), stride=(1,)) #Recreate first layer of the model
    model = torch.load(os.path.join(output_directory,'model_CNN.pth'),map_location=torch.device('cpu')) #Load training parameters

    weight  = model[list(model.keys())[0]]
    bias  = model[list(model.keys())[1]]
    new_state_dict = {'weight': weight, 'bias': bias}
    model_CNN.load_state_dict(new_state_dict)

    motifs = np.zeros((n_filters, 4, kernel_size))
    nsites = np.zeros(n_filters)

    print('Making motifs')

    for i in range(0, len(seq), 100): #Batches of 100
        x = x_test[i:i+100]
        y = model_CNN(x) #Compute output of 1st Convolutional layer
        max_values = torch.max(y,dim=2).values #Compute max value of the sequence for each sample within a layer
        max_indx = torch.max(y,dim=2).indices #Compute max value an obtain indices

        for m in range(n_filters): #Loop through filters
            for n in range(len(x)): #Loop through samples
                # Forward strand
                if max_values[n, m] > 0: #If the values is activated i.e. we are applying a ReLu function
                    motif =  x[n, :, max_indx[n, m]:max_indx[n, m] + kernel_size]  #Add this motif in the filter row
                    nsites[m] += 1     #Add one point to that filter
                    motifs[m] += motif.numpy()


    x = motifs.reshape(filter_size,kernel_size,4)

    for i_filter in range(x.shape[0]):
      x_filter = x[i_filter]

      #Normalize matrix
      PWM = x_filter / x_filter.sum(axis=1)[:, np.newaxis]

      PWM[np.isnan(PWM)] = 0.25 #If any value is Nan is because the whole position was 0, hence all nucleotides have the same probability (?)
      #Run sequence logo
      ppm = seqlogo.Ppm(PWM)
      seqlogo.seqlogo(ppm, ic_scale = True, filename=os.path.join(output_directory,'seqLOGO_filter'+str(i_filter)+'.png'), format = 'png', size = 'medium')


    #Save motifs in MEME format
    motifs_file = open(os.path.join(output_directory,'PFM_motifs.meme'), 'w')
    motifs_file.write('MEME version 4.9.0\n\n'
                      'ALPHABET= ACGT\n\n'
                      'strands: + -\n\n'
                      'Background letter frequencies (from uniform background):\n'
                      'A 0.25000 C 0.25000 G 0.25000 T 0.25000\n\n')

    for m in range(n_filters):
        if nsites[m] == 0:
            continue
        motifs_file.write('MOTIF M%i O%i\n' % (m, m))
        motifs_file.write("letter-probability matrix: alength= 4 w= %i nsites= %i E= 1337.0e-6\n" % (kernel_size, nsites[m]))
        for j in range(kernel_size):
            motifs_file.write("%f %f %f %f\n" % tuple(1.0 * motifs[m, 0:4, j] / np.sum(motifs[m, 0:4, j])))
        motifs_file.write('\n')

    motifs_file.close()

def main(args=False):
    "Perform CNN for SuRE data."
    if not args: args = parse_args()

    #ARGUMENTS

    global main_directory, file_testset, output_directory
    main_directory = args.main_directory
    file_testset = args.file_testset
    output_directory = args.output_directory

    #Load test SET

    x_test = load_test_data(file_testset)

    obtain_PPM(n_filters=128, kernel_size=10, x_test=x_test)


def parse_args():
    "Parses inputs from commandline and returns them as a Namespace object."

    parser = ArgumentParser(prog = 'python3 test.py',
        formatter_class = RawTextHelpFormatter)

    # Arguments for the input files
    parser.add_argument('main_directory', help='directory with input files (raw table)')

    parser.add_argument('file_testset', help='.gz file name')

    parser.add_argument('output_directory',  help='path to a directory where output files are saved')


    return parser.parse_args()



if __name__ == '__main__':
	main()
