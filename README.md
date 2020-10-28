# DOUBLER Source Code

This repository contains code for the paper [DOUBLER: Unified Representation Learning of Biological Entities and Documents for Predicting Protein-Disease Relationships](http://biorxiv.org/content/10.1101/2020.10.27.357202v1) (Sztyler & Malone, 2020).

If you have any questions, feel free to reach out!

 
## Dependencies
All dependencies are installed with ``pip install -r requirements.txt``. Please ensure that `pip` and `setuptools` are up-to-date by running `pip install --upgrade pip setuptools`. We recommend Python 3.7 to avoid any conflicts.

## DOUBLER
The implementation of DOUBLER is located in `doubler.py`.

## Training & Evaluation
To run training and evaluation, execute `python runner.py`. Hyperparameters can be adapted within the `main()` method of the respective file. Please note that you first have to unzip the file `features_bow.zip` in the folder `/data`. Be aware that it will grow to almost 700 MB.

## Source databases and their licenses
An example dataset is part of this repository. In particular, we used the following datasets:

| Source type                                | Source name                                  | License                                         |
|--------------------------------------------|----------------------------------------------|-------------------------------------------------|
| edge (gene-disease), freetext (disease)    | [DisGeNet](https://www.disgenet.org/)[1]     | CC BY-NC-CA                                     |
| edge (gene-gene)                           | [STRING](https://string-db.org/)[2]          | CC BY                                           |
| structured annotations (genes)             | [GO](http://geneontology.org/)[3]            | CC BY                                           |
| structured annotations (disease-phenotype) | [HPO](https://hpo.jax.org/app/)[4]           | Custom: [HPO](https://hpo.jax.org/app/license)  |
| free text (genes)                          | [MyGene](https://mygene.info/)[5]            | Apache License 2.0                              |

## References
[1] Lo Surdo, P., Calderone, A., Iannuccelli, M., Licata, L., Peluso, D., Castagnoli, L., Cesareni, G., and Perfetto, L. (2017). DISNOR: A disease network open resource. Nucleic Acids Research, 46(D1), D527–D534.  
[2] Szklarczyk, D., Gable, A. L., Lyon, D., Junge, A., Wyder, S., Huerta-Cepas, J., Simonovic, M., Doncheva, N. T., Morris, J. H., Bork, P., et al. (2018). STRING v11: protein–protein association networks with increased coverage, supporting functional discovery in genome-wide experimental datasets. Nucleic Acids Research, 47(D1), D607–D613.  
[3] Gene Ontology Consortium (2004). The Gene Ontology (GO) database and informatics resource. Nucleic Acids Research, 32(suppl_1), D258–D261  
[4] Köhler, S., Doelken, S. C., Mungall, C. J., Bauer, S., Firth, H. V., Bailleul-Forestier, I., Black, G. C., Brown, D. L., Brudno, M., Campbell, J., et al. (2014). The Human Phenotype Ontology project: Linking molecular biology and disease through phenotype data. Nucleic Acids Research, 42(D1), D966–D974.  
[5] Wu, C., MacLeod, I., and Su, A. I. (2013). BioGPS and MyGene.info: Organizing online, gene-centric information. Nucleic Acids Research, 41(D1), D561–D565.