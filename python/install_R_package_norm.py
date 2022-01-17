import rpy2.robjects.packages as rpackages
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)
utils.install_packages("norm")
DirichletReg = rpackages.importr("norm")
