rm(list=ls()) # clear out the environment

require(caret)
require(pROC)
require(corrplot)

source("src.r")
source("kmeans_filter.R")

# ---------------------------------------------------------------------------
# line up the data

dat.s = read.csv('/Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/R code and files/Superior_June2023.csv')
dat.i = read.csv('/Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/R code and files/Inferior_June2023.csv')
dim(dat.s)
dim(dat.i)
y.sup = as.factor(dat.s$Adherent_invasive)
x.sup = dat.s[,-c(1:3)]
y.inf = as.factor(dat.i$Adherent_invasive)
x.inf = dat.i[,-c(1:3)]
levels(y.sup) = c("No","Yes")
levels(y.inf) = c("No","Yes")
y.sup = relevel(y.sup,ref="Yes")
y.inf = relevel(y.inf,ref="Yes")

if(!identical(dat.s$Study_ID,dat.i$Study_ID)){stop("ID's not aligned.")} 
if(!identical(y.sup,y.inf)){stop("The Y's are not aligned.")}

# ---------------------------------------------------------------------------
# preprocessing

x.sup.full = x.sup
x.inf.full = x.inf
# filtering
cutoff = 0.8
x.sup = cor.filter(nzv.filter(x.sup),cutoff=cutoff)
x.inf = cor.filter(nzv.filter(x.inf),cutoff=cutoff)

# filter out MRI-sensitive features
xmri.s = data.frame(MRI.scanner=dat.s$MRI.scanner,x.sup)
xmri.i = data.frame(MRI.scanner=dat.i$MRI.scanner,x.inf)
mri.s = rmv.mri.sensitive(x=xmri.s)
mri.i = rmv.mri.sensitive(x=xmri.i)
x.sup = x.sup[,-mri.s$sensitive.set]
x.inf = x.inf[,-mri.i$sensitive.set]

# common set of features
nms.sup = names(x.sup)
nms.inf = names(x.inf)
common.features = intersect(nms.sup,nms.inf)
length(common.features)

# if we want to plot the correlation matrix:
if(0){
	M = M0 = cor(x.sup) # or x.inf
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
# Univariate analyses

# Mann-Whitney tests
uv.tests.sup = uv.mw(x=x.sup,y=y.sup) # smallest p=0.6126
uv.tests.inf = uv.mw(x=x.inf,y=y.inf) # smallest p=0.7054

# Predictive performance...
# ... from superior ROI:
uv.s = uv.analyses(x=x.sup,y=y.sup)
# check distributions
summary(apply(uv.s$sens,2,mean))
summary(apply(uv.s$spec,2,mean))
summary(apply(uv.s$aucs,2,mean))
summary(apply(uv.s$accs,2,mean))
# ... from inferior ROI:
uv.i = uv.analyses(x=x.inf,y=y.inf)
# check distributions
summary(apply(uv.i$sens,2,mean))
summary(apply(uv.i$spec,2,mean))
summary(apply(uv.i$aucs,2,mean))
summary(apply(uv.i$accs,2,mean))

# save output
save.image("univariate_output.RData")

# ---------------------------------------------------------------------------
# Multivariate analyses

# bootstrap multivariate models
mv.sup = mv.analyses(x=x.sup,y=y.sup)
mv.inf = mv.analyses(x=x.inf,y=y.inf)

# save all
save.image(file="script_run.Rdata")
