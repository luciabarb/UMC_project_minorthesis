Prediction of variant effects on promoter activity with deep learning-based sequence models

Non-coding DNA comprises 98% of the human genome, where the majority of mutations in cancer take place. Most non-coding regions contain the information to regulate gene expression but only
recently have we started studying the impact of mutations in these regions. Therefore, there is an urgent need to characterize the regulatory genome and its functional variants. Current com-
putational algorithms that aim to unravel this matter make use of epigenetic marks measured on a genome-wide scale. Hence, their performance is limited by linkage disequilibrium (LD), as it is
difficult to distinguish whereas a trait is caused by local or distant single-nucleotide polymorphisms (SNPs). SuRE technology solves this issue by quantitatively measuring the promoter activity of
a single genomic fragment. Thus, by using deep learning models in combination with the SuRE dataset, we constructed a convolutional neural network (CNN) that is able to predict promoter
activity by detecting known protein-binding motifs in HEPG2 and K562 cell lines. Although the current model cannot predict variant effects, further improvements ensure a promising future to
the combinations of SuRE dataset and deep learning models to decode the non-coding genome.


In these repository you will find the code for training and tunning the hyperparemeters of a Convolutional neural network (CNN) (final_CNN_GLM_SuRE_tunning.py), along with the motif extraction of the first convolutional layer (final_conv1d_PWM.py).
