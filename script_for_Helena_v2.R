rm(list=ls()) # clear out the environment

require(caret)
require(pROC)
require(corrplot)

# ---------------------------------------------------------------------------
# line up the data

source("make_data.R")
source("src.r")
source("kmeans_filter.R")

# if you want to plot the correlation matrix:

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
	# plot.cor(mat,tl.col='grey30',addgrid.col='lightgrey',method='ellipse')
	# plot.cor(mat,tl.col='grey30',addgrid.col='lightgrey',method='shade')

# ---------------------------------------------------------------------------
# Univariate models

# Mann-Whitney tests w.r.t. y
p = ncol(x)
p.values = NA*numeric(p)
names(p.values) = names(x)
for(i in 1:p){
	xi = x[,i]
	t.out = wilcox.test(xi~y,conf.int=TRUE)
	pval = t.out$p.value
	p.values[i] = pval
}
p.values.adj = p.adjust(p.values,method="fdr")
pdf("mann_whitney_tests.pdf")
par(mfrow=c(4,2),mar=c(4,2,3,1))
COLS = c('pink','cyan')
for(i in 1:p){
	xi = x[,i]
	boxplot(xi~y,names=levels(y),col=COLS,
		xlab="", ylim=c(0,1),
		main=paste(names(x)[i],round(p.values.adj[i],4),sep="\np="))
}
dev.off()
# sort variables by decreasing p-value:
sort(p.values.adj,decreasing=TRUE) 

# identifying discrepancies between scanner types
yAB = x$MRI.scanner
x$MRI.scanner = NULL
p = ncol(x)
p.values.AB = NA*numeric(p)
names(p.values.AB) = names(x)
for(i in 1:p){
	xi = x[,i]
	pval = wilcox.test(xi~yAB)$p.value
	p.values.AB[i] = pval
}
p.values.AB.adj = p.adjust(p.values.AB,method="fdr")
pdf("mann_whitney_MRI_scanner.pdf")
par(mfrow=c(4,2),mar=c(4,2,3,1))
COLS = c('pink','cyan')
for(i in 1:p){
	xi = x [,i]
	boxplot(xi~yAB,names=levels(yAB),col=COLS,
	xlab="", ylim=c(0,1),
	main=paste(names(x)[i],round(p.values.AB.adj[i],4),sep="\np="))
}
dev.off()
# sort variables by decreasing p-value:
sort(p.values.AB.adj,decreasing=TRUE) 
alpha=0.05
i.rm = which(p.values.AB.adj< alpha)
x = x[,-i.rm]
ncol(x)
length(i.rm)

# ---------------------------------------------------------------------------
# bootstrap univariate LR models

B = 100
SEED = 4060
set.seed(SEED)
res = matrix(NA,nrow=ncol(x),ncol=2)
ci.l = matrix(NA,nrow=ncol(x),ncol=2)
ci.u = matrix(NA,nrow=ncol(x),ncol=2)
aucs = accs = matrix(NA,nrow=B,ncol=ncol(x))
for(i in 1:ncol(x)){
	lro = blr.uv(x[,i],y)
	accs[,i] = lro$acc
	aucs[,i] = lro$auc
	res[i,] = apply(lro,2,mean)
	ci.l[i,] = apply(lro,2,quantile,0.025)
	ci.u[i,] = apply(lro,2,quantile,0.975)
}
colnames(res) = names(lro)
rownames(res) = names(x)
ci.acc = cbind(ci.l[,1], ci.u[,1])
ci.auc = cbind(ci.l[,2], ci.u[,2])
res.acc = round(cbind(res[,1], ci.acc),3)
res.auc = round(cbind(res[,2], ci.auc),3)
colnames(res.acc) = c("Acc","2.5%","97.5%")
colnames(res.auc) = c("AUC","2.5%","97.5%")
# inspect output looking for top predictors
cbind(sort(res[,2]))
# max accs and aucs
apply(res,2,max)
# Top 10 accuracies:
sort(res[,1],decreasing=TRUE)[1:10]
# Top 10 AUCs:
sort(res[,2],decreasing=TRUE)[1:10]
# which uv model is best:
apply(res,2,which.max)
i.opt = which.max(res[,"auc"])
res[i.opt,]
names(x)[i.opt]
finals = data.frame(acc=accs[,i.opt],auc=aucs[,i.opt])
boxplot(finals,  ylim=c(0,1))


# ---------------------------------------------------------------------------
# multivariate bootstrap models

B = 100
SEED = 6040
set.seed(SEED)

# build models...
# LASSO:
bo.rglm = train.class(x,y,method='glmnet',
					adjust = 0.5,
					preProc=c("center","scale"))
# RF:
bo.rf = train.class(x,y,method='rf',tG=expand.grid(mtry=c(8,10,12)))
# kNN:
bo.knn = train.class(x,y,method='knn',
					preProc=c("center","scale"))
# polynomial SVM:
bo.svm = train.class(x,y,method='svmPoly',
					preProc=c("center","scale"))

# prediction performance
res = rbind(bo.rglm$res,bo.rf$res,bo.knn$res,bo.svm$res)
row.names(res) = c("R.GLM","RF","KNN","SVM")
round(res,3)*100

# variable importance
(vi.rglm = varImp(bo.rglm$model.roc))
(vi.rf = varImp(bo.rf$model.roc))
(vi.knn = varImp(bo.knn$model.roc))
(vi.svm = varImp(bo.svm$model.roc))

# retrieve important sets...
vi.co = 55 # lower cut-off on variable importance
rglm.set = which(vi.rglm$importance>vi.co)
rf.set = which(vi.rf$importance>vi.co)
knn.set = which(vi.knn$importance[,2]>vi.co)
svm.set = which(vi.svm$importance[,2]>vi.co)

x.rglm = x[,rglm.set]
x.rf = x[,rf.set]
x.knn = x[,knn.set]
x.svm = x[,svm.set]
c(ncol(x), ncol(x.rglm), ncol(x.rf), ncol(x.knn), ncol(x.svm))

# ... and inspect in terms of PCA clusters
# PCA of feature matrix fed into all models:
pca = pca.kmeans.filter(x,k=6)
pca = pca.kmeans.filter(x,k=12)
# summary(pca$pca)
# plot(pca$pca)
pve = round(summary(pca$pca)$importance[3,]*100,1)
barplot(pve,las=2,main="% variance explained")
abline(h=c(90,95),lwd=2)
(pca.clusters = cbind(sort(pca$cluster)))
cluster.and.vimp(x.rglm,vi.rglm)
# what each model picks:
cluster.and.vimp(x.rf,vi.rf)
cluster.and.vimp(x.knn,vi.knn)
cluster.and.vimp(x.svm,vi.svm)

labs = pca$cluster
cols = c(1:max(labs))[labs]
par(pty='s')
pca.biplot(pca,rglm.set)
pca.biplot(pca,rf.set)
pca.biplot(pca,knn.set)
pca.biplot(pca,svm.set)
pca.biplot(pca)

# RFE for random forest?
# (will need to code this up to include more results if works well)
if(0){ # <-- "switch" to 1 to run RF with RFE...
	ctrl <- rfeControl(functions = rfFuncs,
                   method = "boot",
                   number = B,
                   verbose = FALSE)
	subsets <- c(2:5, 10, 15, 20, 25, 30)
	rf.rfe <- rfe(x, y, sizes = subsets, 
					rfeControl = ctrl,
					tuneGrid=expand.grid(mtry=c(10,12)))
	rf.rfe
} # all models with 10+ predictors seem similar on reduced set

# tables of results, with CI's :
summ.tb(bo.rglm$model.roc$resample,bo.rglm$model.acc$resample)
# or to stack them up by model (with CI's):
results = round(rbind(
summ.tb.in.line(bo.rglm$model.roc$resample,bo.rglm$model.acc$resample),
summ.tb.in.line(bo.rf$model.roc$resample,bo.rf$model.acc$resample),
summ.tb.in.line(bo.knn$model.roc$resample,bo.knn$model.acc$resample),
summ.tb.in.line(bo.svm$model.roc$resample,bo.svm$model.acc$resample)
),3)
# convenient:
write.csv(results,file="results_with_CIs.csv",row.names=F)

# bootstraped ROC plot	
roco.rglm = bs.roc(bo.rglm$model.roc)
roco.rf = bs.roc(bo.rf$model.roc)
roco.knn = bs.roc(bo.knn$model.roc)
roco.svm = bs.roc(bo.svm$model.roc)
par(pty='s')
plot(roco.rglm, col=1, xlim=c(1,0))
plot(roco.rf,col=2,xlim=c(1,0), add=TRUE)
plot(roco.knn,col=4,xlim=c(1,0), add=TRUE)
plot(roco.svm,col='navy',xlim=c(1,0), add=TRUE)
legend("bottomright",col=c(1,2,4,'navy'),lty=1,lwd=3,bty='n',
	legend=row.names(res))
cbind(row.names(res),
	round(c(roco.rglm$auc, roco.rf$auc, roco.knn$auc, roco.svm$auc),3))

par(pty='s',lwd=2)
plot(c(0:1),c(0:1),t='n',xlab="Predicted",ylab="Observed")
abline(a=0,b=1,lwd=.5)
q = seq(0,1,by=.2)
o=calib(mod=bo.rglm,q=q, col=1)
o=calib(mod=bo.rf,q=q,col=2,add=TRUE)
o=calib(mod=bo.knn,q=q,col=4,add=TRUE)
o=calib(mod=bo.svm,q=q,col='navy',add=TRUE)
legend("bottomright",col=c(1,2,4,'navy'),lty=1,lwd=3,bty='n',
       legend=row.names(res))

# this bit generates a pdf file of graphical summaries of performance

	resamps <- resamples(list(R.GLM = bo.rglm$model.roc,
                          RF = bo.rf$model.roc,
                          KNN = bo.knn$model.roc,
                          SVM = bo.svm$model.roc))
	theme1 <- trellis.par.get()
	theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
	theme1$plot.symbol$pch = 16
	theme1$plot.line$col = rgb(1, 0, 0, .7)
	theme1$plot.line$lwd <- 2
	pdf(file="output.pdf")
	#
	trellis.par.set(theme1)
	bwplot(resamps, layout = c(3, 1))
	trellis.par.set(caretTheme())
	dotplot(resamps, metric = "ROC")
	trellis.par.set(theme1)
	xyplot(resamps, what = "BlandAltman")
	splom(resamps)
	#
	difValues <- diff(resamps)
	difValues
	summary(difValues)
	trellis.par.set(theme1)
	bwplot(difValues, layout = c(3, 1))
	#
	dev.off()

# For more:
# https://topepo.github.io/caret/measuring-performance.html
	
	
	