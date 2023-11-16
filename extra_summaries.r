rm(list=ls())
load(file="out/univariate_analyses.Rdata")

# ---------------------------------------------------
# PAS classification plot
# ---------------------------------------------------
y.inf = as.factor(pool.inf$Adherent_invasive)
y.sup = as.factor(pool.sup$Adherent_invasive)
levels(y.inf) = c("No","Yes")
levels(y.sup) = c("No","Yes")
y.inf = relevel(y.inf,ref="Yes")
y.sup = relevel(y.sup,ref="Yes")

pas.pca <- function(x,y,i=1,j=2){
	s = apply(x,2,sd)
	x = x[,-which(s<0.01)]
	pca = prcomp(x,scale=TRUE)
	par(font=2,font.lab=2,font.axis=2)
	x1 = pca$x[,i]/pca$sdev[i]
	x2 = pca$x[,j]/pca$sdev[j]
	cols = c(2,4)[y]
	pchs = c('Yes','No')[y]
	plot(x1,x2,pch=pchs,col=cols,xlab="PC1",ylab="PC2",
		main="PCA-based classification")
	abline(h=0,col=8)
	abline(v=0,col=8)
}
i=1 # which PC to use as x-axis
j=2 # which PC to use as y-axis
pas.pca(x=x.sup,y=y.sup,i=i,j=j)

# NB: you can subset the data to only consider "better" 
# features as follows (can change aucs to sens and .6 to other):
ij = which(apply(uva.pooled.inf$aucs,2,mean)>.6)
pas.pca(x=x.sup[,ij],y=y.sup,i=i,j=j)

pas.pca <- function(x,y,i=1,j=2){
  s = apply(x,2,sd)
  if(any(s<0.01)){x = x[,-which(s<0.01)]}
  pca = prcomp(x,scale=TRUE)
  par(font=2,font.lab=2,font.axis=2)
  x1 = pca$x[,i]/pca$sdev[i]
  x2 = pca$x[,j]/pca$sdev[j]
  cols = c(2,4)[y]
  pchs = c('Yes','No')[y]
  plot(x1,x2,pch=pchs,col=cols,xlab="PC1",ylab="PC2",
       main="PCA-based classification")
  abline(h=0,col=8)
  abline(v=0,col=8)
  props = round(100*cumsum(pca$sdev)/sum(pca$sdev),1)
  print(cat("Proportion of variance explained by (PC1,PC2):",props[1],"%\n"))
  return(props)
}

# example
x = iris[,1:4]
y = as.factor(c('No','Yes')[as.numeric(iris$Species=='setosa')+1])
o = pas.pca(x,y)
o[1]
pas.pca(x=x.sup[,ij],y=y.sup,i=i,j=j)
x = x=x.sup[,ij]
y = y=y.sup
o = pas.pca(x,y)
o[2]
# ---------------------------------------------------
# Univariate analyses
# ---------------------------------------------------

# #1 for univariate analyses
summ <- function(uva,THR=.65){
	cat("Metrics greater than",(THR*100),"%, out of ",ncol(uva$aucs),"features:\n")
	cat("AUC:", round(mean(uva$aucs>=THR)*100,2), "%\n")
	cat("Accuracy:", round(mean(uva$accs>=THR)*100,2), "%\n")
	cat("Sensitivity:", round(mean(uva$sens>=THR)*100,2), "%\n")
	cat("Specificity:", round(mean(uva$spec>=THR)*100,2), "%\n")
	cat("Both sens + spec:", round(mean((uva$sens>=THR)&(uva$spec>=THR))*100,2), "%\n")
}
summ(uva.pooled.sup)
summ(uva.pooled.inf)

# Overall summary tables
summ.tab <- function(uva,THR=.65){
	so = function(v){ 
		pTHR = mean(uva$aucs>=THR)
		o = c(summary(v), pTHR) 
		return(round(o*100,2))
	}
	aucs = apply(uva$aucs,2,mean)
	accs = apply(uva$accs,2,mean)
	sens = apply(uva$sens,2,mean)
	spec = apply(uva$spec,2,mean)
	df = cbind(so(aucs),so(accs),so(sens),so(spec))
	row.names(df)[7] = paste("%>",THR,sep='')
	colnames(df) = c("AUC","Accuracy","Sensitivity","Specificity")
	return(df)
}
# Example:
summ.tab(uva.pooled.sup)
summ.tab(uva.pooled.inf)
# ---------------------------------------------------
# Multivariate analyses
# ---------------------------------------------------

summ.mv <- function(o,THR=.65){
	accs = o$model.acc$resample$Accuracy
	aucs = o$model.roc$resample$ROC	
	sens = o$model.roc$resample$Sens
	spec = o$model.roc$resample$Spec
	cat("Metrics greater than",(THR*100),"%:\n")
	cat("AUC:", round(mean(aucs>=THR)*100,2), "%\n")
	cat("Accuracy:", round(mean(accs>=THR)*100,2), "%\n")
	cat("Sensitivity:", round(mean(sens>=THR)*100,2), "%\n")
	cat("Specificity:", round(mean(spec>=THR)*100,2), "%\n")
	cat("Both sens + spec:", round(mean((sens>=THR)&(spec>=THR))*100,2), "%\n")
}

fd = paste("out/run_5000_hilo_",SEED,"_R30k3.Rdata",sep=""))
load(fd)
colnames(res.ho) = c("auc","acc","sens","spec")
rk = order(res.ho[,"sens"],decreasing=TRUE)
i = rk[1]
o = oz[[i]] 
summ.mv(o)

# ---------------------------------------------------
# Heatmaps
# ---------------------------------------------------

# #2 Heatmap (change x.sup to x.inf and y.sup to y.inf if required)

PAS.heatmap <- function(x,y,mute=TRUE,...){
	require(superheat)
	minmax.scale <- function(v){
		return((v-min(v))/(max(v)-min(v)))	
	}
	lu <- function(v){ length(unique(v)) }
	s = apply(x,2,lu)
	ij = which(s<10)
	if(mute & length(ij)){ x = x[,-ij] }
	x = apply(x,2,minmax.scale)
	o = superheat(X=x,membership.rows=y,pretty.order.cols=TRUE,...)
	return(list(o=o,x=x,s=s,ejects=ij))
}
o = PAS.heatmap(x.sup, y.sup)
# NB: o = PAS.heatmap(x.sup[,185:194], y.sup)
o = PAS.heatmap(x.inf, y.inf)

# NB: you can subset the data to only consider "better" 
# features as follows (can change aucs to sens and .6 to other):
ij = which(apply(uva$aucs,2,mean)>.6)
o = PAS.heatmap(x.sup[,ij], y.sup)
