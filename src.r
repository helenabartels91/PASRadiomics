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

train.class <- function(x,y,method='glmnet',tG=NULL,adjust=1,...){
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
	if(method=='glmnet'){
		tG$lambda = adjust*tG$lambda
	}
	bo.acc = train(x,y,method=method,metric="Accuracy",
					trControl=trC.acc,tuneGrid=tG,...)
	bo.final = train(x,y,method=method,metric="ROC",
					trControl=trC.final,tuneGrid=tG,...)
	res = lapply(bo$resample[,1:3],mean)
	res$Acc = bo.acc$results$Accuracy
	return(list(res=data.frame(res),	
				model.roc=bo,model.acc=bo.acc,model.final=bo.final))
}

bs.roc <- function(model){
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

blr.uv <- function(xi,y,cutoff=.5){
# bootstrapping univariate logisic regression
	n = length(y)
	acc = auc = numeric(B)
	for(b in 1:B){
		ib = sample(1:n,n,replace=TRUE)
		uib = unique(ib)	
		x.b = xi[ib]
		y.b = y[ib]
		x.oob = xi[-uib]
		y.oob = y[-uib]
		glmo = glm(y.b~x.b,family="binomial")
		po.score = predict(glmo,data.frame(x.b=x.oob))
		po.prob = predict(glmo,data.frame(x.b=x.oob),type='response')
		po.class = as.factor((po.prob>cutoff))
		levels(po.class) = levels(y)
		cm = caret::confusionMatrix(po.class,ref=y.oob)
		roco = roc(response=y.oob,predictor=po.prob,quiet=TRUE)
		acc[b] = cm$overall[1]
		auc[b] = roco$auc
	}
	return(data.frame(acc,auc))
}

blr.mv <- function(xi,y,cutoff=.5){
# bootstrapping multivariate logisic regression
	n = length(y)
	acc = auc = numeric(B)
	for(b in 1:B){
		ib = sample(1:n,n,replace=TRUE)
		uib = unique(ib)	
		x.b = xi[ib,]
		y.b = y[ib]
		x.oob = xi[-uib,]
		y.oob = y[-uib]
		glmo = glm(y.b~.,data=x.b,family="binomial")
		po.score = predict(glmo,x.oob)
		po.prob = predict(glmo,x.oob,type='response')
		po.class = as.factor((po.prob>cutoff))
		levels(po.class) = levels(y)
		cm = caret::confusionMatrix(po.class,ref=y.oob)
		roco = roc(response=y.oob,predictor=po.prob,quiet=TRUE)
		acc[b] = cm$overall[1]
		auc[b] = roco$auc
	}
	return(data.frame(acc,auc))
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
	mean.kpis = apply(resamp[,1:3],2,mean)
	se.kpis = apply(resamp[,1:3],2,sd)
	ci.kpis = t(apply(resamp[,1:3],2,quantile,c(.025,.975)))
	return(cbind(mean.kpis,ci.kpis))
}
summ.acc <- function(resamp){
# here resamp = bo.rglm$model.acc$resample
	mean.kpis = mean(resamp$Acc)
	se.kpis = sd(resamp$Acc)
	ci.kpis = quantile(resamp$Acc,c(.025,.975))
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

binary.calibration.curve <- function(predicted,observed,Lref=1,
			q, lower=NULL, upper=NULL,
			x.lab="Predicted", y.lab="Observed",
			add=FALSE, ...){
# Groups into quintiles by default.
# - predicted are estimated probabilities P(Y=1|X)
# - observed are event indicator values within {0;1} 
# 
	gp = cut(predicted,q,labels=F)
	nj = table(gp)
	G = max(gp,na.rm=T)
	means.p = means.o = numeric(G)
	sd.o = sd.p = numeric(G)
	for(g in 1:G){
		ig = which(gp==g)
		means.p[g] = mean(predicted[ig])
		means.o[g] = mean(observed[ig]==Lref)
		sd.o[g] = sd(observed[ig]==Lref)
		sd.p[g] = sd(predicted[ig])
	}
	if(any(is.nan(means.p))){
		ii = which(is.nan(means.p))
		means.p[ii] = NA
		means.o[ii] = NA
	}
	nina = which(!is.na(means.p))
	points(means.p[nina],means.o[nina],pch=20,t='b',...)
	pm.o = pmax(means.o[nina]-1.96*sd.o[nina]/sqrt(nj),0)
	pM.o = pmin(means.o[nina]+1.96*sd.o[nina]/sqrt(nj),1)
	# pm.o = pmax(means.o[nina]-1.96*sd.p[nina]/sqrt(nj))
	# pM.o = pmin(means.o[nina]+1.96*sd.p[nina]/sqrt(nj))
	segments(x0=means.p[nina],y0=pm.o,y1=pM.o,col=8,lwd=.6)
	return(list(mean.p=means.p,mean.o=means.o))
}

calib <- function(mod,q=seq(0,1,by=.2),add=FALSE,...){
	preds = predict(mod$model.roc)
	probs = predict(mod$model.roc,type="prob")[,1]
	o = binary.calibration.curve(predicted=probs,observed=y,Lref=levels(y)[1],q=q,add=add,...)
	return(o)
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
