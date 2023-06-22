# Eric Wolsztynski, University College Cork, 2023
# eric.w@ucc.ie

# commodity functions

step.lr <- function(xr,y,metric=c("acc","auc")){
	metric = metric[1]
	p = ncol(xr)
	step.crit = step.acc  = step.auc = numeric(p-1)
	set = NULL
	cands = xr
	for(i in 1:(p-1)){
		acc.j = auc.j = numeric(ncol(cands))
		if(!is.null(ncol(cands))){
			for(j in 1:ncol(cands)){
				set.seed(4060)
				modj = c(set,names(cands)[j])
				lro = blr(x=xr[,modj],y)
				auc.j[j] = mean(lro$auc)
				acc.j[j] = mean(lro$acc)
			}
		}
		if(metric=="acc"){
			crit = acc.j
		} else {
			crit = auc.j	
		}
		j.opt = which.max(crit)
		step.crit[i] = max(crit)
		step.acc[i] = acc.j[j.opt]
		step.auc[i] = auc.j[j.opt]
		set = c(set,names(cands)[j.opt])	
		cands = cands[,-j.opt]
	}
	i.opt = which.max(step.crit)
	fmod = set[1:i.opt]
	return(list(step.crit=step.crit,step.acc=step.acc,
			step.auc=step.auc,metric=metric,
			i.opt=i.opt,final.model=fmod))
}

train.class <- function(x,y,method='glmnet',tG=NULL,B,...){
# re-using train() to get various metrics together	
	trC = trainControl(method="boot",number=B,
		summaryFunction=twoClassSummary,
		classProbs=TRUE,savePredictions="final",...)
	trC.acc = trainControl(method="boot",number=B,
		classProbs=TRUE,savePredictions="final",...)
	trC.final = trainControl(method="none",
		summaryFunction=twoClassSummary,
		classProbs=TRUE,savePredictions="final",...)
	bo = train(x,y,method=method,tuneGrid=tG,
					trControl=trC,metric="ROC", ...)
	tG = expand.grid(bo$bestTune)
	bo.acc = train(x,y,method=method,metric="Accuracy",
					trControl=trC.acc,tuneGrid=tG,...)
	bo.final = train(x,y,method=method,metric="ROC",
					trControl=trC.final,tuneGrid=tG,...)
	res = lapply(bo$resample[,1:3],mean)
	res$Acc = bo.acc$results$Accuracy
	return(list(res=data.frame(res),	
				model.roc=bo,model.acc=bo.acc,model.final=bo.final))
}

bs.roc <- function(model,y){
# bootstrap ROC curve from averaged train() OOB predictions
	# calculate bootstrapped P(Y='Yes'|X)
	n = nrow(x)
	pm = numeric(n)
	for(i in 1:n){
		ix = which(model$pred$rowIndex==i)
		pm[i] = mean(model$pred$Yes[ix])
	}
	# corresponding ROC analysis
	roco = roc(response=y, predictor=pm, levels=c('No','Yes'))
	return(roco)
}

plot.cor <- function(x,NC=32,...){
# customizing the correlation heatmap
	require(corrplot)
	# tmwr_cols <- colorRampPalette(c("blue","red"))
	tmwr_cols <- colorRampPalette(c("blue","lightblue",
					"white","tomato","red"))
	ccols = tmwr_cols(NC)
	corrplot(x, col=ccols, ...)
}

blr.uv <- function(xi,y,B,cutoff=.5){
# bootstrapping univariate logisic regression
	n = length(y)
	acc = auc = sens = spec = numeric(B)
	for(b in 1:B){
		ib = sample(1:n,n,replace=TRUE)
		uib = unique(ib)	
		x.b = xi[ib]
		y.b = y[ib]
		x.oob = xi[-uib]
		y.oob = y[-uib]
		while(any(table(y.oob)==0)){ # just re-draw
			ib = sample(1:n,n,replace=TRUE)
			uib = unique(ib)	
			x.b = xi[ib]
			y.b = y[ib]
			x.oob = xi[-uib]
			y.oob = y[-uib]			
		}
		glmo = glm(y.b~x.b,family="binomial")
		po.score = predict(glmo,data.frame(x.b=x.oob))
		po.prob = predict(glmo,data.frame(x.b=x.oob),type='response')
		po.class = as.factor(as.numeric(po.prob>cutoff))
		levels(po.class) = c("No","Yes")
		cm = caret::confusionMatrix(po.class,ref=y.oob,positive="Yes")
		roco = roc(response=y.oob,predictor=po.prob,quiet=TRUE)
		acc[b] = cm$overall[1]
		auc[b] = roco$auc
		sens[b] = cm$byClass[1]
		spec[b] = cm$byClass[2]
	}
	return(data.frame(acc,auc,sens,spec))
}

blr.mv <- function(xi,y,B,cutoff=.5){
# bootstrapping multivariate logisic regression
	n = length(y)
	acc = auc = sens = spec = numeric(B)
	for(b in 1:B){
		ib = sample(1:n,n,replace=TRUE)
		uib = unique(ib)	
		x.b = xi[ib,]
		y.b = y[ib]
		x.oob = xi[-uib,]
		y.oob = y[-uib]
		while(any(table(y.oob)==0)){ # just re-draw
			ib = sample(1:n,n,replace=TRUE)
			uib = unique(ib)	
			x.b = xi[ib]
			y.b = y[ib]
			x.oob = xi[-uib]
			y.oob = y[-uib]			
		}
		glmo = glm(y.b~.,data=x.b,family="binomial")
		po.score = predict(glmo,x.oob)
		po.prob = predict(glmo,x.oob,type='response')
		po.class = as.factor((po.prob>cutoff))
		levels(po.class) = c("No","Yes")
		cm = caret::confusionMatrix(po.class,ref=y.oob,positive="Yes")
		roco = roc(response=y.oob,predictor=po.prob,quiet=TRUE)
		acc[b] = cm$overall[1]
		auc[b] = roco$auc
		sens[b] = cm$byClass[1]
		spec[b] = cm$byClass[2]
	}
	return(data.frame(acc,auc,sens,spec))
}

blr <- function(xi,y,cutoff=.5){
	if(!is.null(dim(xi))){
		return(blr.mv(xi,y,cutoff))
	} else {
		return(blr.uv(xi,y,cutoff))
	}
}

boot.ml <- function(zx,x,xm,y,B=100,rl.lam.cv=0.04,rr.lam.cv=0.04){
# bootstrapping a bunch of classifiers
	require(randomForest)
	n = nrow(x)
	fits.rlcox = fits.rrcox = fits.bcox = fits.rf = NULL
	preds.rlcox = matrix(NA,nrow=n,ncol=B)
	preds.rrcox = matrix(NA,nrow=n,ncol=B)
	preds.bcox = matrix(NA,nrow=n,ncol=B)
	preds.rf = matrix(NA,nrow=n,ncol=B)
	c.rlcox = c.rrcox = c.bcox = c.rf = c.rf.getcindex = numeric(B)
	rl.lam.cvs = rr.lam.cvs = numeric(B)
	for(b in 1:B){
		if(b%%10==0){ cat("Loop",b,"... \n") }
		ib = sample(1:n,n,replace=TRUE)
		uib = unique(ib)
		
		x.b = x[ib,]
		xm.b = xm[ib,]
		y.b = y[ib]
		x.oob = x[-uib,]
		xm.oob = xm[-uib,]
		y.oob = y[-uib]
		
		# regularized (penalized) Cox
		if(is.null(rl.lam.cv)){
			rl.lam.cv = cv.glmnet(x=xm.b, y=y.b, family="cox")$lambda.min
		}
		rl.lam.cvs[b] = rl.lam.cv
		rlcox.b = glmnet(x=xm.b, y=y.b, family="cox", lambda=rl.lam.cv)
		rlcox.oob = predict(rlcox.b,newx=xm.oob)[,1]
		preds.rlcox[-uib,b] = rlcox.oob
		# NB: to obtain predicted relative risk instead of linear predictors:
		# hr.rcox.oob = predict(cr,newx=xm.oob,type="response")[,1]
		# see how they relate: plot(exp(rcox.oob),hr.rcox.oob); abline(a=0,b=1)
	
		# ridge Cox
		if(is.null(rr.lam.cv)){
			rr.lam.cv = cv.glmnet(x=xm.b, y=y.b, family="cox", alpha=0)$lambda.min
		}
		rr.lam.cvs[b] = rr.lam.cv
		rrcox.b = glmnet(x=xm.b, y=y.b, family="cox", lambda=rr.lam.cv, alpha=0)
		rrcox.oob = predict(rrcox.b,newx=xm.oob)[,1]
		preds.rrcox[-uib,b] = rrcox.oob

		# boosted Cox
		zx.b = zx[ib,]
		zx.oob = zx[-uib,]
		bcox.b = gbm(Surv(surtim,surind)~., data=zx.b, distribution="coxph")
		bcox.oob = predict(bcox.b, newdata=zx.oob, n.trees=bcox.b$n.trees)
		preds.bcox[-uib,b] = bcox.oob
	
		# random forest
		rfo = randomForest(y~., data=zx.b) 
		rf.oob = predict(rfo,newdata=zx.oob)
		preds.rf[-uib,b] = rf.oob$predicted
		
		fits.rlcox[[b]] = rlcox.b
		fits.rrcox[[b]] = rrcox.b
		fits.bcox[[b]] = bcox.b
		fits.rf[[b]] = rfo
		
		# performance
		c.rlcox[b] = Cindex(rlcox.oob, y=y.oob) 
		c.rrcox[b] = Cindex(rrcox.oob, y=y.oob) 
		c.bcox[b] = Cindex(bcox.oob, y=y.oob) 
		c.rf[b] = Cindex(rf.oob$predicted,y=y.oob)
	}
	p.df = array(NA,dim=c(n,B,4))
	p.df[,,1] = preds.rlcox
	p.df[,,2] = preds.rrcox
	p.df[,,3] = preds.bcox
	p.df[,,4] = preds.rf
	c.df = cbind(c.rlcox, c.rrcox, c.bcox, c.rf)
	models = c("LASSO Cox", "Ridge Cox", "Boosted Cox","RF")
	colnames(c.df) = models
	return(list(preds=p.df, c.indexes=c.df, c.rf.getcindex= c.rf.getcindex,
				rr.lam.cvs=rr.lam.cvs, rl.lam.cvs=rl.lam.cvs, 
				fits.rlcox=fits.rlcox, fits.rrcox=fits.rrcox, fits.bcox=fits.bcox, fits.rf=fits.rf))
}

summ.roc <- function(resamp){
# here resamp = bo.rglm$model.roc$resample
	mean.kpis = apply(resamp[,1:3],2,mean,na.rm=T)
	se.kpis = apply(resamp[,1:3],2,sd,na.rm=T)
	ci.kpis = t(apply(resamp[,1:3],2,quantile,c(.025,.975),na.rm=T))
	return(cbind(mean.kpis,ci.kpis))
}
summ.acc <- function(resamp){
# here resamp = bo.rglm$model.acc$resample
	mean.kpis = mean(resamp$Acc,na.rm=T)
	se.kpis = sd(resamp$Acc,na.rm=T)
	ci.kpis = quantile(resamp$Acc,c(.025,.975),na.rm=T)
	return(c(Acc=mean.kpis,ci.kpis))
}
summ.tb <- function(resamp.roc, resamp.acc){
	rbind(summ.roc(resamp.roc),summ.acc(resamp.acc))	
}
summ.tb.in.line <- function(resamp.roc, resamp.acc){
	ro = c(t(summ.roc(resamp.roc)),t(summ.acc(resamp.acc)))
	nms = c(rownames(summ.roc(resamp.roc)),"Acc")
	nms.025=paste(nms,"2.5%",sep="-")
	nms.975=paste(nms,"97.5%",sep="-")
	names(ro) = c(t(cbind(nms,nms.025,nms.975)))
	return(ro)
}
calibration.curve <- function(predicted,observed,
			lower=NULL, upper=NULL,
			x.lab="Predicted",y.lab="Observed",
			q=seq(0,1,by=.2),seg.col=1,...){
# Group into quintiles by default.
# observed, lower, and upper are averages from estimates, 
# this could be reconsidered.
# 
	qs = quantile(predicted,q)
	gp = cut(predicted,qs,labels=F)
	G = max(gp,na.rm=T)
	means.p = means.o = numeric(G)
	pm.o = pM.o = sd.o = numeric(G)
	L2 = levels(observed)[2]
	for(g in 1:G){
		ig = which(gp==g)
		means.p[g] = mean(predicted[ig]==L2)
		means.o[g] = mean(observed[ig]==L2)
		sd.o[g] = sd(observed[ig]==L2)
	}
	par(pty='s')
	plot(c(0:1),c(0:1),t='n',xlab=x.lab,ylab=y.lab)
	abline(a=0,b=1,lwd=.5)
	points(means.p,means.o,pch=20,t='b',...)
	segments(x0=means.p,y0=pm.o,y1=pM.o,col=seg.col)
}

calib <- function(mod){
	preds = predict(mod$model.roc)
	probs = predict(mod$model.roc,type="prob")[,1]
	calibration.curve(predicted=probs,observed=y)
}

cluster.and.vimp <- function(x.in,vi.in){
	io = order(pca.clusters[names(x.in),])
	nms.in = names(x.in)[io]
	df = data.frame(cluster=pca.clusters[nms.in,],
				importance=vi.in$importance[nms.in,1])
	return(df)
}

pca.biplot <- function(pca,subset,a=1,b=2,...){
	R = pca$pca$rotation[subset,]
	rr = apply(R,2,range)
	plot(R[,c(a,b)],pch='',xlim=c(rr[,a]),ylim=c(rr[,b]),...)
	abline(h=0,lty=3,col=1,lwd=.5)
	abline(v=0,lty=3,col=1,lwd=.5)
	for(i in 1:nrow(R)){
		arrows(0,0,R[i,a],R[i,b], col=cols[i], length=.08)
	}
	labs = pca$cluster
	fake.names = paste("X",1:nrow(R),sep='')
	cols = c(1:max(labs))[labs]
	text(R[,c(a,b)],labels=fake.names,col=cols)
}

clean.dataset <- function(dat){
	nms = names(dat)
	ij = c(grep("Study_ID",nms),
		grep("Study.ID",nms),
		grep("Image$",nms),
		grep("Mask$",nms),
		grep("diagnostics_Image.",nms),
		grep("diagnostics_Mask.original_",nms),
		grep("diagnostics_Configuration_",nms),
		grep("diagnostics_Versions",nms),
		grep("Hysterectomy_Yes.1.",nms),		
		grep("RCC_Yes.1.",nms),		
		grep("Configuration_Settings",nms),
		grep("original_Size",nms),		
		grep("interpolated_Size",nms),		
		grep("Hash",nms),		
		grep("Dimensionality",nms),		
		grep("BoundingBox",nms),		
		grep("Spacing",nms),		
		grep("VolumeNum",nms),		
		grep("CenterOfMass",nms),		
		grep("MassIndex",nms))
	if(length(ij)){
		return(dat[,-ij])
	} else {
		return(dat)
	}
}

# pre-filtering
nzv.filter <- function(dat){
	ic = nearZeroVar(dat)
	if(length(ic)){ dat = dat[,-ic]	}
	return(dat)
}
cor.filter <- function(dat,cutoff=0.8){
	M = cor(dat)
	ic = findCorrelation(M, cutoff)
	if(length(ic)){ dat = dat[,-ic]	}
	return(dat)
}
make.data <- function(dat,cutoff=0.8){
	if(!is.null(dat$Study.ID)){
		id = dat$Study.ID
	} else {
		id = dat$Study_ID
	}
	# remove redundant info
	dat = clean.dataset(dat)
	dat = na.omit(dat)
	# target variable
	y = as.factor(dat[,"Adherent_invasive"])
	levels(y) = c("No","Yes")
	dat$Adherent_invasive = NULL
	# filters
	dat.f = nzv.filter(dat)
	dat.f = cor.filter(dat.f,cutoff=cutoff)
	x = dat.f
	return(list(x=x,y=y,id=id))
}
make.test.data <- function(dat){
	if(!is.null(dat$Study.ID)){
		id = dat$Study.ID
	} else {
		id = dat$Study_ID
	}
	# remove redundant info
	dat = na.omit(dat)
	# target variable
	y = as.factor(dat[,"Adherent_invasive"])
	levels(y) = c("No","Yes")
	dat$Adherent_invasive = NULL
	x = dat
	return(list(x=x,y=y,id=id))
}

# univariate Mann-Whitney tests 
uv.mw <- function(x,y,doplot=FALSE){
	p = ncol(x)
	p.values = NA*numeric(p)
	names(p.values) = names(x)
	for(i in 1:p){
		xi = x[,i]
		t.out = wilcox.test(xi~y,conf.int=TRUE,exact=F)
		pval = t.out$p.value
		p.values[i] = pval
	}
	p.values.adj = p.adjust(p.values,method="fdr")
	if(doplot){
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
	}
	# sort variables by decreasing p-value:
	return(sort(p.values.adj,decreasing=TRUE))
}

# identifying discrepancies between scanner types
rmv.mri.sensitive <- function(x,doplot=FALSE){
	yAB = x$MRI.scanner
	x$MRI.scanner = NULL
	p = ncol(x)
	p.values.AB = NA*numeric(p)
	names(p.values.AB) = names(x)
	for(i in 1:p){
		xi = x[,i]
		pval = wilcox.test(xi~yAB,exact=F)$p.value
		p.values.AB[i] = pval
	}
	p.values.AB.adj = p.adjust(p.values.AB,method="fdr")
	if(doplot){
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
	}
	alpha=0.05
	i.rm = which(p.values.AB.adj< alpha)
	if(length(i.rm)){ x = x[,-i.rm] }
	return(list(x=x,sensitive.set=i.rm))
}

uv.analyses <- function(x,y,B=100,SEED=4060){
	if(!is.factor(y)){ y = as.factor(y) }
	set.seed(SEED)
	res = matrix(NA,nrow=ncol(x),ncol=4)
	ci.l = matrix(NA,nrow=ncol(x),ncol=4)
	ci.u = matrix(NA,nrow=ncol(x),ncol=4)
	aucs = accs = matrix(NA,nrow=B,ncol=ncol(x))
	sens = spec = matrix(NA,nrow=B,ncol=ncol(x))
	for(i in 1:ncol(x)){
		lro = blr.uv(x[,i],y,B)
		accs[,i] = lro$acc
		aucs[,i] = lro$auc
		sens[,i] = lro$sens
		spec[,i] = lro$spec
		res[i,] = apply(lro,2,mean)
		ci.l[i,] = apply(lro,2,quantile,0.025)
		ci.u[i,] = apply(lro,2,quantile,0.975)
	}
	colnames(sens) = names(x)
	colnames(spec) = names(x)
	colnames(accs) = names(x)
	colnames(aucs) = names(x)
	return(list(aucs=aucs,accs=accs,sens=sens,spec=spec,
				res=res,ci.l=ci.l,ci.u=ci.u))
}

mv.analyses <- function(x,y,B=100,SEED=6040){
	set.seed(SEED)
	# build models...
	# LASSO:
	bo.rglm = train.class(x,y,method='glmnet',B=B,
						# adjust = 0.5,
						preProc=c("center","scale"))
	# RF:
	bo.rf = train.class(x,y,method='rf',tG=expand.grid(mtry=c(8,10,12)),B=B)
	# kNN:
	bo.knn = train.class(x,y,method='knn',B=B,
						preProc=c("center","scale"))
	# polynomial SVM:
	bo.svm = train.class(x,y,method='svmPoly',B=B,
						preProc=c("center","scale"))
	return(list(bo.rglm=bo.rglm,bo.rf=bo.rf,bo.knn=bo.knn,bo.svm=bo.svm))
}

mv.results <- function(bo.rglm,bo.rf,bo.knn,bo.svm){
	o.auc = c(mean(bo.rglm$model.roc$resample$ROC),
				mean(bo.rf$model.roc$resample$ROC),
				mean(bo.knn$model.roc$resample$ROC),
				mean(bo.svm$model.roc$resample$ROC,na.rm=T))
	o.acc = c(mean(bo.rglm$model.acc$resample$Accuracy),
				mean(bo.rf$model.acc$resample$Accuracy),
				mean(bo.knn$model.acc$resample$Accuracy),
				mean(bo.svm$model.acc$resample$Accuracy))
	o.sens = c(mean(bo.rglm$model.roc$resample$Sens),
				mean(bo.rf$model.roc$resample$Sens),
				mean(bo.knn$model.roc$resample$Sens),
				mean(bo.svm$model.roc$resample$Sens))
	o.spec = c(mean(bo.rglm$model.roc$resample$Spec),
				mean(bo.rf$model.roc$resample$Spec),
				mean(bo.knn$model.roc$resample$Spec),
				mean(bo.svm$model.roc$resample$Spec,na.rm=T))
	out = cbind(o.auc,o.acc,o.sens,o.spec)
	rownames(out) = c("RGLM","RF","kNN","SVM")
	colnames(out) = c("AUC","Accuracy","Sensitivity","Specificity")
	return(out)	
}
