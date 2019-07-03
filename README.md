# LKDL-Method
This repo is dedicated to testing of method presented in Linearized Kernel Dictionary Learning by Alona Golts and Michael Elad

Experiments so far don't actually do any dictionary learning. Signal subspace matrices are simply formed by taking k/2 principal eigenvectors from mapped H0 and H1 samples and an MSC type classifier is tested. Later updates will likely include experiments when subspace matrices are instead dictionaries learned via some commonly used DL and estimates are generated via some sparse coding rather than LS like we do in MSC.

Referenced Paper: https://arxiv.org/pdf/1509.05634.pdf
