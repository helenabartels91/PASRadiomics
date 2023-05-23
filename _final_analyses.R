# Eric Wolsztynski, University College Cork, 2023
# eric.w@ucc.ie
rm(list=ls()) # clear out the environment

require(caret)
require(pROC)
require(corrplot)

load("script_run.Rdata")

# -----------------------------------------------------------
# univariate analyses 

colnames(uv.sup$sens) = colnames(uv.sup$spec) = colnames(uv.sup$aucs)
# accuracies superior ROI
m.accs.sup = apply(uv.sup$accs,2,mean)
cbind(mean=m.accs.sup, t(apply(uv.sup$accs,2,quantile,c(0.025,0.975))))
# AUCs superior ROI
m.aucs.sup = apply(uv.sup$aucs,2,mean)
cbind(mean=m.aucs.sup, t(apply(uv.sup$aucs,2,quantile,c(0.025,0.975))))
# top feature superior ROI
m.accs.sup[which.max(m.accs.sup)]
m.aucs.sup[which.max(m.aucs.sup)]
# sensitivity superior ROI
m.sens.sup = apply(uv.sup$sens,2,mean)
ci.sens.sup = t(apply(uv.sup$sens,2,quantile,c(0.025,0.975)))
cbind(mean=m.sens.sup,ci.sens.sup)
m.sens.sup[which.max(m.sens.sup)]
# specificity superior ROI
m.spec.sup = apply(uv.sup$spec,2,mean)
ci.spec.sup = t(apply(uv.sup$spec,2,quantile,c(0.025,0.975)))
cbind(mean=m.spec.sup,ci.spec.sup)
m.spec.sup[which.max(m.spec.sup)]
#
# Top 10 AUC+Sens superior
sort(m.aucs.sup,decreasing=TRUE)[1:10]
sort(m.sens.sup,decreasing=TRUE)[1:10]
boxplot(uv.sup$sens)
IQR.sens.sup = apply(t(apply(uv.sup$sens,2,quantile,c(0.25,0.75))),1,diff)
IQR.spec.sup = apply(t(apply(uv.sup$spec,2,quantile,c(0.025,0.975))),1,diff)

colnames(uv.inf$sens) = colnames(uv.inf$spec) = colnames(uv.inf$aucs)
# accuracies inferior ROI
m.accs.inf = apply(uv.inf$accs,2,mean)
cbind(mean=m.accs.inf, t(apply(uv.inf$accs,2,quantile,c(0.025,0.975))))
# AUCs inferior ROI
m.aucs.inf = apply(uv.inf$aucs,2,mean)
cbind(mean=m.aucs.inf, t(apply(uv.inf$aucs,2,quantile,c(0.025,0.975))))
# top feature inferior ROI
m.accs.inf[which.max(m.accs.inf)]
m.aucs.inf[which.max(m.aucs.inf)]
# sensitivity inferior ROI
m.sens.inf = apply(uv.inf$sens,2,mean)
ci.sens.inf = t(apply(uv.inf$sens,2,quantile,c(0.025,0.975)))
cbind(mean=m.sens.inf,ci.sens.inf)
m.sens.inf[which.max(m.sens.inf)]
# specificity inferior ROI
m.spec.inf = apply(uv.inf$spec,2,mean)
ci.spec.inf = t(apply(uv.inf$spec,2,quantile,c(0.025,0.975)))
cbind(mean=m.spec.inf,ci.spec.inf)
m.spec.inf[which.max(m.spec.inf)]

# superior vs inferior MW tests on common features
svi.inf = svi.sup = numeric(length(common.features))
svi.cors = numeric(length(common.features))
for(i in 1:length(common.features)){
	ff = common.features[i]
	# correlations
	svi.cors[i] = cor(uv.sup$accs[,ff],uv.inf$accs[,ff])
	# svi.sup corresponds to H_a: ACC_sup > ACC_inf
	svi.sup[i] = wilcox.test(uv.sup$accs[,ff],uv.inf$accs[,ff],alt='g')$p.value
	svi.inf[i] = wilcox.test(uv.sup$accs[,ff],uv.inf$accs[,ff],alt='l')$p.value
}
# no alignment between inferior and superior radiomics
summary(svi.cors[-which(svi.cors>.4)]) 
common.features[which(svi.cors>.4)]
boxplot(svi.cors) 
common.features[which(svi.cors==1)] # these should correlate to 1!

svi.sup.a = p.adjust(svi.sup,method='fdr')
svi.inf.a = p.adjust(svi.inf,method='fdr')
data.frame(svi.sup.a,row.names=common.features)
mean(svi.sup.a<0.05) 
mean(svi.inf.a<0.05) 
common.features[which(svi.sup.a<0.05)]
common.features[which(svi.inf.a<0.05)]

par(mfrow=c(2,1),mar=c(12,2,3,1))
#
pick = which(apply(uv.sup$aucs,2,median)>0.7)
boxplot(uv.sup$aucs[,pick],main="AUC superior")
pick = which(apply(uv.inf$aucs,2,median)>0.7)
boxplot(uv.inf$aucs[,pick],las=2,main="AUC inferior")
#
pick = which(apply(uv.sup$accs,2,median)>0.6)
boxplot(uv.sup$accs[,pick],main="Accuracies superior")
pick = which(apply(uv.inf$accs,2,median)>0.6)
boxplot(uv.inf$accs[,pick],las=2,main="Accuracies inferior")

# colnames(res) = names(lro)
# rownames(res) = names(x)
# ci.acc = cbind(ci.l[,1], ci.u[,1])
# ci.auc = cbind(ci.l[,2], ci.u[,2])
# res.acc = round(cbind(res[,1], ci.acc),3)
# res.auc = round(cbind(res[,2], ci.auc),3)
# colnames(res.acc) = c("Acc","2.5%","97.5%")
# colnames(res.auc) = c("AUC","2.5%","97.5%")
# # inspect output looking for top predictors
# cbind(sort(res[,2]))
# # max accs and aucs
# apply(res,2,max)
# # Top 10 accuracies:
# sort(res[,1],decreasing=TRUE)[1:10]
# # Top 10 AUCs:
# sort(res[,2],decreasing=TRUE)[1:10]
# # which uv model is best:
# apply(res,2,which.max)
# i.opt = which.max(res[,"auc"])
# res[i.opt,]
# names(x)[i.opt]
# finals = data.frame(acc=accs[,i.opt],auc=aucs[,i.opt])
# boxplot(finals,  ylim=c(0,1))
# xlim=c(1,0)

# -----------------------------------------------------------
# multivariate analyses 

names(mv.sup)
names(mv.sup$bo.svm)
# hist(mv.sup$bo.svm$model.roc$resample$ROC)
# hist(mv.inf$bo.svm$model.roc$resample$ROC)

# overall results
# Superior:
round(mv.results(mv.sup$bo.rglm,mv.sup$bo.rf,mv.sup$bo.knn,mv.sup$bo.svm),3)
# Inferior:
round(mv.results(mv.inf$bo.rglm,mv.inf$bo.rf,mv.inf$bo.knn,mv.inf$bo.svm),3)

# sup ROC
wilcox.test(mv.sup$bo.rglm$model.roc$resample$ROC,mu=0.5,alt='g')$p.value
wilcox.test(mv.sup$bo.rf$model.roc$resample$ROC,mu=0.5,alt='g')$p.value
wilcox.test(mv.sup$bo.knn$model.roc$resample$ROC,mu=0.5,alt='g')$p.value
wilcox.test(mv.sup$bo.svm$model.roc$resample$ROC,mu=0.5,alt='g')$p.value
# sup Sens
wilcox.test(mv.sup$bo.rglm$model.roc$resample$Sens,mu=0.5,alt='g')$p.value
wilcox.test(mv.sup$bo.rf$model.roc$resample$Sens,mu=0.5,alt='g')$p.value
wilcox.test(mv.sup$bo.knn$model.roc$resample$Sens,mu=0.5,alt='g')$p.value
wilcox.test(mv.sup$bo.svm$model.roc$resample$Sens,mu=0.5,alt='g')$p.value
# inf ROC
wilcox.test(mv.inf$bo.rglm$model.roc$resample$ROC,mu=0.5,alt='g')$p.value
wilcox.test(mv.inf$bo.rf$model.roc$resample$ROC,mu=0.5,alt='g')$p.value
wilcox.test(mv.inf$bo.knn$model.roc$resample$ROC,mu=0.5,alt='g')$p.value
wilcox.test(mv.inf$bo.svm$model.roc$resample$ROC,mu=0.5,alt='g')$p.value
# inf Sens
wilcox.test(mv.inf$bo.rglm$model.roc$resample$Sens,mu=0.5,alt='g')$p.value
wilcox.test(mv.inf$bo.rf$model.roc$resample$Sens,mu=0.5,alt='g')$p.value
wilcox.test(mv.inf$bo.knn$model.roc$resample$Sens,mu=0.5,alt='g')$p.value
wilcox.test(mv.inf$bo.svm$model.roc$resample$Sens,mu=0.5,alt='g')$p.value

# superior vs inferior AUCs, H_A: AUC_sup != AUC_inf
wilcox.test(mv.sup$bo.rglm$model.roc$resample$ROC,
	mv.inf$bo.rglm$model.roc$resample$ROC, alt='two')$p.value
wilcox.test(mv.sup$bo.rf$model.roc$resample$ROC,
	mv.inf$bo.rf$model.roc$resample$ROC, alt='two')$p.value
wilcox.test(mv.sup$bo.knn$model.roc$resample$ROC,
	mv.inf$bo.knn$model.roc$resample$ROC, alt='two')$p.value
wilcox.test(mv.sup$bo.svm$model.roc$resample$ROC,
	mv.inf$bo.svm$model.roc$resample$ROC, alt='two')$p.value
#
# superior vs inferior AUCs, H_A: AUC_sup > AUC_inf
wilcox.test(mv.sup$bo.rglm$model.roc$resample$ROC,
	mv.inf$bo.rglm$model.roc$resample$ROC, alt='g')$p.value
wilcox.test(mv.sup$bo.rf$model.roc$resample$ROC,
	mv.inf$bo.rf$model.roc$resample$ROC, alt='g')$p.value
wilcox.test(mv.sup$bo.knn$model.roc$resample$ROC,
	mv.inf$bo.knn$model.roc$resample$ROC, alt='g')$p.value
wilcox.test(mv.sup$bo.svm$model.roc$resample$ROC,
	mv.inf$bo.svm$model.roc$resample$ROC, alt='g')$p.value
# 
# superior vs inferior accuracies, H_A: Acc_inf > Acc_sup 
wilcox.test(mv.sup$bo.rglm$model.acc$resample$Accuracy,
	mv.inf$bo.rglm$model.acc$resample$Accuracy, alt='l')$p.value
wilcox.test(mv.sup$bo.rf$model.acc$resample$Accuracy,
	mv.inf$bo.rf$model.acc$resample$Accuracy, alt='l')$p.value
wilcox.test(mv.sup$bo.knn$model.acc$resample$Accuracy,
	mv.inf$bo.knn$model.acc$resample$Accuracy, alt='l')$p.value
wilcox.test(mv.sup$bo.svm$model.acc$resample$Accuracy,
	mv.inf$bo.svm$model.acc$resample$Accuracy, alt='l')$p.value
# 
# superior vs inferior accuracies, H_A: Sens_sup > Sens_inf
wilcox.test(mv.sup$bo.rglm$model.roc$resample$Sens,
	mv.inf$bo.rglm$model.roc$resample$Sens, alt='g')$p.value
wilcox.test(mv.sup$bo.rf$model.roc$resample$Sens,
	mv.inf$bo.rf$model.roc$resample$Sens, alt='g')$p.value
wilcox.test(mv.sup$bo.knn$model.roc$resample$Sens,
	mv.inf$bo.knn$model.roc$resample$Sens, alt='g')$p.value
wilcox.test(mv.sup$bo.svm$model.roc$resample$Sens,
	mv.inf$bo.svm$model.roc$resample$Sens, alt='g')$p.value
# 
# superior vs inferior accuracies, H_A: Sens_inf > Sens_sup
wilcox.test(mv.sup$bo.rglm$model.roc$resample$Sens,
	mv.inf$bo.rglm$model.roc$resample$Sens, alt='l')$p.value
wilcox.test(mv.sup$bo.rf$model.roc$resample$Sens,
	mv.inf$bo.rf$model.roc$resample$Sens, alt='l')$p.value
wilcox.test(mv.sup$bo.knn$model.roc$resample$Sens,
	mv.inf$bo.knn$model.roc$resample$Sens, alt='l')$p.value
wilcox.test(mv.sup$bo.svm$model.roc$resample$Sens,
	mv.inf$bo.svm$model.roc$resample$Sens, alt='l')$p.value

# overall view
dev.new()
par(mfcol=c(4,4),mar=c(2,1,4,1))
#
boxplot(mv.sup$bo.rf$model.acc$resample$Accuracy,
	mv.inf$bo.rf$model.acc$resample$Accuracy, 
	names=c('Superior','Inferior'),ylim=c(0,1),
	col=c('cyan','pink'), main='RF accuracy')
boxplot(mv.sup$bo.rglm$model.acc$resample$Accuracy,
	mv.inf$bo.rglm$model.acc$resample$Accuracy, 
	names=c('Superior','Inferior'),ylim=c(0,1),
	col=c('cyan','pink'), main='RGLM accuracy')
boxplot(mv.sup$bo.knn$model.acc$resample$Accuracy,
	mv.inf$bo.knn$model.acc$resample$Accuracy, 
	names=c('Superior','Inferior'),ylim=c(0,1),
	col=c('cyan','pink'), main='kNN accuracy')
boxplot(mv.sup$bo.svm$model.acc$resample$Accuracy,
	mv.inf$bo.svm$model.acc$resample$Accuracy, 
	names=c('Superior','Inferior'),ylim=c(0,1),
	col=c('cyan','pink'), main='SVM accuracy')
#
boxplot(mv.sup$bo.rf$model.roc$resample$ROC,
	mv.inf$bo.rf$model.roc$resample$ROC, 
	names=c('Superior','Inferior'),ylim=c(0,1),
	col=c('cyan','pink'), main='RF AUC')
boxplot(mv.sup$bo.rglm$model.roc$resample$ROC,
	mv.inf$bo.rglm$model.roc$resample$ROC, 
	names=c('Superior','Inferior'),ylim=c(0,1),
	col=c('cyan','pink'), main='RGLM AUC')
boxplot(mv.sup$bo.knn$model.roc$resample$ROC,
	mv.inf$bo.knn$model.roc$resample$ROC, 
	names=c('Superior','Inferior'),ylim=c(0,1),
	col=c('cyan','pink'), main='kNN AUC')
boxplot(mv.sup$bo.svm$model.roc$resample$ROC,
	mv.inf$bo.svm$model.roc$resample$ROC, 
	names=c('Superior','Inferior'),ylim=c(0,1),
	col=c('cyan','pink'), main='SVM AUC')
#
boxplot(mv.sup$bo.rf$model.roc$resample$Sens,
	mv.inf$bo.rf$model.roc$resample$Sens, 
	names=c('Superior','Inferior'),ylim=c(0,1),
	col=c('cyan','pink'), main='RF sensitivity')
boxplot(mv.sup$bo.rglm$model.roc$resample$Sens,
	mv.inf$bo.rglm$model.roc$resample$Sens, 
	names=c('Superior','Inferior'),ylim=c(0,1),
	col=c('cyan','pink'), main='RGLM sensitivity')
boxplot(mv.sup$bo.knn$model.roc$resample$Sens,
	mv.inf$bo.knn$model.roc$resample$Sens, 
	names=c('Superior','Inferior'),ylim=c(0,1),
	col=c('cyan','pink'), main='kNN sensitivity')
boxplot(mv.sup$bo.svm$model.roc$resample$Sens,
	mv.inf$bo.svm$model.roc$resample$Sens, 
	names=c('Superior','Inferior'),ylim=c(0,1),
	col=c('cyan','pink'), main='SVM sensitivity')
#
boxplot(mv.sup$bo.rf$model.roc$resample$Spec,
	mv.inf$bo.rf$model.roc$resample$Spec, 
	names=c('Superior','Inferior'),ylim=c(0,1),
	col=c('cyan','pink'), main='RF specificity')
boxplot(mv.sup$bo.rglm$model.roc$resample$Spec,
	mv.inf$bo.rglm$model.roc$resample$Spec, 
	names=c('Superior','Inferior'),ylim=c(0,1),
	col=c('cyan','pink'), main='RGLM specificity')
boxplot(mv.sup$bo.knn$model.roc$resample$Spec,
	mv.inf$bo.knn$model.roc$resample$Spec, 
	names=c('Superior','Inferior'),ylim=c(0,1),
	col=c('cyan','pink'), main='kNN specificity')
boxplot(mv.sup$bo.svm$model.roc$resample$Spec,
	mv.inf$bo.svm$model.roc$resample$Spec, 
	names=c('Superior','Inferior'),ylim=c(0,1),
	col=c('cyan','pink'), main='SVM specificity')

# # prediction performance
# res = rbind(bo.rglm$res,bo.rf$res,bo.knn$res,bo.svm$res)
# row.names(res) = c("R.GLM","RF","KNN","SVM")
# round(res,3)*100

# --------------------------------------------------------

# flick between one or the other: 
mv.mod = mv.sup; x = x.sup; y = y.sup
# mv.mod = mv.inf; x = x.inf; y = y.inf
# mv.results2(mv.sup, x = x.sup, y = y.sup)

# mv.results2 <- function(mv.mod,x,y){
	# variable importance
	(vi.rglm = varImp(mv.mod$bo.rglm$model.roc))
	(vi.rglm = varImp(mv.mod$bo.rglm$model.roc))
	(vi.rf = varImp(mv.mod$bo.rf$model.roc))
	(vi.knn = varImp(mv.mod$bo.knn$model.roc))
	(vi.svm = varImp(mv.mod$bo.svm$model.roc))
	
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
	summ.tb(mv.mod$bo.rglm$model.roc$resample,mv.mod$bo.rglm$model.acc$resample)
	# or to stack them up by model (with CI's):
	results = round(rbind(
	summ.tb.in.line(mv.mod$bo.rglm$model.roc$resample,mv.mod$bo.rglm$model.acc$resample),
	summ.tb.in.line(mv.mod$bo.rf$model.roc$resample,mv.mod$bo.rf$model.acc$resample),
	summ.tb.in.line(mv.mod$bo.knn$model.roc$resample,mv.mod$bo.knn$model.acc$resample),
	summ.tb.in.line(mv.mod$bo.svm$model.roc$resample,mv.mod$bo.svm$model.acc$resample)
	),3)
	# convenient:
	write.csv(results,file="results_with_CIs.csv",row.names=F)
	
	# bootstraped ROC plot	
	roco.rglm = bs.roc(mv.mod$bo.rglm$model.roc,y)
	roco.rf = bs.roc(mv.mod$bo.rf$model.roc,y)
	roco.knn = bs.roc(mv.mod$bo.knn$model.roc,y)
	roco.svm = bs.roc(mv.mod$bo.svm$model.roc,y)
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
	o=calib(mod=mv.mod$bo.rglm,q=q, col=1)
	o=calib(mod=mv.mod$bo.rf,q=q,col=2,add=TRUE)
	o=calib(mod=mv.mod$bo.knn,q=q,col=4,add=TRUE)
	o=calib(mod=mv.mod$bo.svm,q=q,col='navy',add=TRUE)
	legend("bottomright",col=c(1,2,4,'navy'),lty=1,lwd=3,bty='n',
	       legend=row.names(res))
	
	# this bit generates a pdf file of graphical summaries of performance
	
		resamps <- resamples(list(R.GLM = mv.mod$bo.rglm$model.roc,
	                          RF = mv.mod$bo.rf$model.roc,
	                          KNN = mv.mod$bo.knn$model.roc,
	                          SVM = mv.mod$bo.svm$model.roc))
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
# }
