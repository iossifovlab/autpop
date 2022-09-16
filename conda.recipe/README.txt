# conda config --set anaconda_upload yes

# to add the tag
git tag -a 0.1 -m 'version 0.1'
git push origin 0.1

To build the package, first, make sure that there are no results stored 
in the demo directory. Then run:

conda build . 

To upload to anaconda:
anaconda upload --user iossifovLab autpop...tar.bz2

