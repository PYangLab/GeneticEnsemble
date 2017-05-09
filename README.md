# GeneticEnsemble

#### Description

Genetic Ensemble is an Java implementation of a genetic algorithm-based ensemble of classifiers designed for feature subset selection https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-11-524. It is a wrapper-based approach that select a subset of features based of their overall performance across a set of classifiers.

This algorithm was originally developed for selecting a panel of interacting SNPs from genome-wide association studies (GWAS), hence the name "GEsnpxPara" where "Para" stands for the implementation of parallel computing version of the algorithm. Please refer to the original repository for more detail on how to apply this for SNP selection [link](https://code.google.com/archive/p/genetic-ensemble-snpx/). Nevertheless, the implemented software package is flexiable and can be applied for gene or protein subset selection from microarray or proteomics data.

#### Reference

Yang, P., Ho, J., Zomaya, A., Zhou, B. (2010). A genetic ensemble approach for gene-gene interaction identification. BMC Bioinformatics, 11(524), 1-15. [[Fulltext](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-11-524)]
