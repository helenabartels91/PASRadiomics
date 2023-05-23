rm(list=ls()) # clear out the environment

require(caret)
require(pROC)
require(corrplot)

source("src.r")
source("kmeans_filter.R")

# ---------------------------------------------------------------------------
# line up the data

dat = read.csv('/Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/R code files/Superior_radiomics_features_lebanon.csv')
dat.sup = make.data(dat)
x.sup = dat.sup$x
y.sup = dat.sup$y
x.sup.full = x.sup

dat = read.csv('/Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/Lebanon data/R code files/Inferior_radiomics_features_lebanon.csv')
dat.inf = make.data(dat)
x.inf = dat.inf$x
y.inf = dat.inf$y
x.inf.full = x.inf

# check alignment
ch1 = cbind(dat.sup$id[-dat.sup$i.co], dat.inf$id[-dat.inf$i.co])
for(i in 1:nrow(ch1)){
	if(!identical(ch1[i,1],ch1[i,2])){ print(paste("ID's not aligned @ row",i)) }
}
if(!identical(dat.sup$y,dat.inf$y)){ stop("The Y's are not aligned.") }
# grep("lbp.3D.k_glrlm_RunLengthNonUniformity",names(x.sup.full))
# grep("lbp.3D.k_glrlm_RunLengthNonUniformity",names(x.inf.full))
# grep("wavelet.HHH_glrlm_LowGrayLevelRunEmphasis",names(x.sup.full))
# grep("wavelet.HHH_glrlm_LowGrayLevelRunEmphasis",names(dat))
# grep("wavelet.HHH_glrlm_LowGrayLevelRunEmphasis",names(x.inf.full))
# # so this one gets filtered out of inferior ROI

dim(x.sup)
dim(x.inf)
nms.sup = names(x.sup)
nms.inf = names(x.inf)
common.features = intersect(nms.sup,nms.inf)
length(common.features)

# if you want to plot the correlation matrix:
if(0){
	M = M0 = cor(x)
	cutoff = 0.5
	diag(M) = 0
	M[which(abs(M)<cutoff)] = 0
	mat = M0 
	#mat = M # <-- comment this line if you want to paint all cells
	colnames(mat) = rownames(mat) = paste("x",c(1:ncol(mat)),sep="")
	# use either of the following:
	# plot.cor(mat,tl.col='grey30',addgrid.col='lightgrey')
	plot.cor(mat,tl.col='grey30',addgrid.col='lightgrey',method='square')
	#complete.obs
	# plot.cor(mat,tl.col='grey30',addgrid.col='lightgrey',method='ellipse')
	# plot.cor(mat,tl.col='grey30',addgrid.col='lightgrey',method='shade')
}

# ---------------------------------------------------------------------------
# Univariate tests

# Mann-Whitney tests
uv.tests.sup = uv.mw(x=x.sup,y=y.sup)
uv.tests.inf = uv.mw(x=x.inf,y=y.inf)

# MRI sensitivity
rm.sup = rmv.mri.sensitive(x.sup)
x.sup = rm.sup$x
x.sup$MRI.scanner = NULL
rm.inf = rmv.mri.sensitive(x.inf)
x.inf = rm.inf$x
x.inf$MRI.scanner = NULL

dim(x.sup)
dim(x.inf)
nms.sup = names(x.sup)
nms.inf = names(x.inf)
common.features = intersect(nms.sup,nms.inf)
length(common.features)

# ---------------------------------------------------------------------------
# bootstrap univariate LR models

uv.sup = uv.analyses(x=x.sup,y=y.sup)
uv.inf = uv.analyses(x=x.inf,y=y.inf)

# ---------------------------------------------------------------------------
# bootstrap multivariate models

mv.sup = mv.analyses(x=x.sup,y=y.sup)
mv.inf = mv.analyses(x=x.inf,y=y.inf)

save.image(file="script_run.Rdata")
