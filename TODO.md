


* Revive the summarize_results.
    * add the global summary table to the autpop if there a model 
    is run more than once.

* (DONE) rename the pU, pC and pD columns. Into what?

* (DONE) split the family_type_key into mom_genotype and dad_genotype

* (DONE) make the FamilyType constructor accept genotypes as strings

* (DONE) Revive save_global_stats_table
    * add precision argument

* (DONE) Add processing and timing messages to standard output

* (DONE) Add the n_processes command line argument
* (DONE) Add the all_families command line argument

* (DONE) Add the following properties to teh family specific stats
    * parents_affected
    * mom_liability
    * dad_liability
    * mom_affected
    * dad_affected

* (DONE) Check, if the paper results are reproduced 
for the precise computation!


* (DONE) Add the following section to the global stats
    Prediction method description
        precice: True or False
        family set description
        number of family types
        number of family types with sampling

* (DONE) If predictions of the first run are precise, don't run more

* (DONE) Implement the family sampling mode