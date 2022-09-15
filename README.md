# autpop

## Introduction

Tool for exploring models of populations with autism.

The tools accompanies our "Wroten et al. Sharing parental 
genomes by siblings concordant or discordant for autism" 
manuscript that will hopefully be published soon. 

The models and the methods used to make predictions about 
populations of families with autism are described in detail in the 
manuscript. 

## Installation

autpop is distributed as a conda package in the iossifovlab conda channel. To install one needs first to intall a conda infrastructure. 
There are many alternative ways to accomplish that. The easiest one may be to install Anaconda from 
https://www.anaconda.com/products/distribution.

Anaconda installs a large number of packages related to Datascience. 
If these are not needed, one can install the much smaller (and quicker to install) Miniconda from https://docs.conda.io/en/latest/miniconda.


Once Anaconda or Miniconda are installed, one can install autpop using the following command prompt command:

```shell
$ conda install -c iossifovlab autpop
```

## Basic use

autpop is a command line tool. To use it one has to define one or more population models in a yaml file. A example of a model definition is:

```yaml
threshold_model:
  name: basic
  male_threshold: 9
  female_threshold: 11
  locus_classes:
  - w: 1
    f: 0.4
    n: 5
  - w: 8
    f: 0.01
    n: 2
```

If we store the above in a file called simple.yaml we can compute the predicted risks and parental genomic sharing by running the following command: 

```shell
$ autpop simple.yaml
```

The intermediate and final results will be stored in three files: family_stats_simple.txt, global_stats_simple.txt and models_results.txt. These files are described in detail in the manuscript.

autpop accepts many optional parameter that controll how the predictions will be computed. For a list and a description of all 
optional parameters one can use:

```shell
$ autpop --help
```




## Soucre code

https://github.com/iossifovlab/autpop

