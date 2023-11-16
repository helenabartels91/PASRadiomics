# Eric Wolsztynski, University College Cork, 2023
# eric.w@ucc.ie

rm(list=ls())
require(caret)

SEEDS = c(1219,1912,4060,4061,6040)
SEED = SEEDS[3]
fd = paste("run_5000_hilo_4060_R30k3_MSIZE_4",sep="")

load(fd)

# tidy up
colnames(res.ho) = c("auc","acc","sens","spec")
rk = order(res.ho[,"sens"],decreasing=TRUE)
res.ho[rk[1:5],]
opt.set = nms[sset[which(sels[rk[1],]==1)]]
print(opt.set)

# overall performances
mean(res.ho[,'auc']>.6) # around 20% 
mean(res.ho[,'auc']>.7) # about 0.5%-0.8% 
# quote CIs instead?--prop of CIs containing [.6,.7] for ex?

i = rk[1]
o = oz[[i]] 
o.acc = mean(o$model.acc$resample$Accuracy)
o.sens = mean(o$model.roc$resample$Sens)
o.spec = mean(o$model.roc$resample$Spec)
o.auc = mean(o$model.roc$resample$ROC)
c(o.auc,o.acc,o.sens,o.spec)
ci.acc = quantile(o$model.acc$resample$Accuracy,c(0.025,0.975))
ci.sens = quantile(o$model.roc$resample$Sens,c(0.025,0.975))
ci.spec = quantile(o$model.roc$resample$Spec,c(0.025,0.975))
ci.auc = quantile(o$model.roc$resample$ROC,c(0.025,0.975))
cis = cbind(ci.auc,ci.acc,ci.sens,ci.spec)
round(t(cis),3)

if(1){ # calibration curve
	o.retrain = train.class(x=x[,opt.set],y=y,model='rf',
		method="repeatedcv",B=30,tG=tG.rf)
	calibration.curve <- function(predicted,observed,
	                              x.lab="Predicted",y.lab="Observed",
	                              q=seq(0,1,by=.2),...){
	  # Group into quintiles by default.
	  # observed are fraction of survived within the groups
	  #
	  dat = data.frame(predicted, observed)
	  qs = quantile(dat$predicted,q)
	  gp = cut(dat$predicted,qs,labels=F)
	  dat$gp <- gp
	  G = max(gp,na.rm=T)
	  means.p = o = numeric(G)
	
	  for(g in 1:G){
	    ig = which(gp==g)
	    means.p[g] = mean(dat$predicted[ig])
	    o[g] = length(which(dat$observed[ig] == 0))/length(dat$gp[ig])
	  }
	  par(pty='s')
	  plot(c(0:1),c(0:1),t='n',xlab=x.lab,ylab=y.lab)
	  abline(a=0,b=1,lwd=.5)
	  points(means.p,o,pch=20,t='b',...)
	}
	mod = o.retrain
	preds = predict(mod$model.roc)
	probs = predict(mod$model.roc,type="prob")[,1]
	par(font.lab=2)
	calibration.curve(predicted=probs,observed=as.numeric(y=="No"),q=seq(0,1,by=.2))
	calibration.curve(predicted=probs,observed=as.numeric(y=="No"),q=c(0,.25,.75,1))
}

# retrain optimal model with clinical covariates
if(1){
	set.seed(SEED)
	clin.set = c('Age','No.of.CS',opt.set)
	o.rerain = train.class(x=x[,clin.set],y=y,model='rf',
		method="repeatedcv",B=30,tG=tG.rf)
	o.acc = mean(o.rerain$model.acc$resample$Accuracy)
	o.sens = mean(o.rerain$model.roc$resample$Sens)
	o.spec = mean(o.rerain$model.roc$resample$Spec)
	o.auc = mean(o.rerain$model.roc$resample$ROC)
	means = c(o.auc,o.acc,o.sens,o.spec)
	o.acc = quantile(o.rerain$model.acc$resample$Accuracy,c(0.025,0.975))
	o.sens = quantile(o.rerain$model.roc$resample$Sens,c(0.025,0.975))
	o.spec = quantile(o.rerain$model.roc$resample$Spec,c(0.025,0.975))
	o.auc = quantile(o.rerain$model.roc$resample$ROC,c(0.025,0.975))
	final.perf = cbind(o.auc,o.acc,o.sens,o.spec)
	res.retrain = cbind(means,t(final.perf))
	round(res.retrain,3)
	
}



# retrain optimal model to evaluate all metrics
if(1){
	set.seed(SEED)
	o.rerain = train.class(x=x[,opt.set],y=y,model='rf',
		method="repeatedcv",B=30,tG=tG.rf)
	o.acc = mean(o.rerain$model.acc$resample$Accuracy)
	o.sens = mean(o.rerain$model.roc$resample$Sens)
	o.spec = mean(o.rerain$model.roc$resample$Spec)
	o.auc = mean(o.rerain$model.roc$resample$ROC)
	c(o.auc,o.acc,o.sens,o.spec)
	o.acc = quantile(o.rerain$model.acc$resample$Accuracy,c(0.025,0.975))
	o.sens = quantile(o.rerain$model.roc$resample$Sens,c(0.025,0.975))
	o.spec = quantile(o.rerain$model.roc$resample$Spec,c(0.025,0.975))
	o.auc = quantile(o.rerain$model.roc$resample$ROC,c(0.025,0.975))
	final.perf = cbind(o.auc,o.acc,o.sens,o.spec)
	round(t(final.perf),3)
}

# compare to glm?
gg = data.frame(alpha=0,lambda=0)
glmo = train.class(x=x[,opt.set],y=y,model='glm',k=k,B=B,
	method="repeatedcv",tG=NULL)
glmo.acc = mean(glmo$model.acc$resample$Accuracy)
glmo.sens = mean(glmo$model.roc$resample$Sens)
glmo.spec = mean(glmo$model.roc$resample$Spec)
glmo.auc = mean(glmo$model.roc$resample$ROC)
glmout = cbind(glmo.auc,glmo.acc,glmo.sens,glmo.spec)

rf.bx = cbind(o$model.roc$resample$ROC,
				o$model.acc$resample$Accuracy,
				o$model.roc$resample$Sens,
				o$model.roc$resample$Spec)
glm.bx = cbind(glmo$model.roc$resample$ROC,
				glmo$model.acc$resample$Accuracy,
				glmo$model.roc$resample$Sens,
				glmo$model.roc$resample$Spec)
colnames(rf.bx) = c('AUC','Accuracy','Sensitivity','Specificity')
colnames(glm.bx) = c('AUC','Accuracy','Sensitivity','Specificity')

par(font=2,font.lab=2,font.axis=2)
boxplot(rf.bx,col='red',ylim=c(0,1))
boxplot(glm.bx,col='cyan',add=TRUE,names=NA)
# better plot
library(ggplot2)
library(reshape2)
rf.df = data.frame(rf.bx,rep("RF",nrow(rf.bx)))
glm.df = data.frame(glm.bx,rep("GLM",nrow(glm.bx)))
names(rf.df)[5] = 'Model'
names(glm.df)[5] = 'Model'
df = rbind(rf.df,glm.df)
df.lg = melt(df,id="Model")
par(font=2,font.lab=2,font.axis=2)
ggplot(df.lg,aes(x=variable,y=value,fill=Model)) + 
	geom_boxplot() + 
	theme(plot.title=element_text(face="bold"),
			axis.text=element_text(face="bold"),
			text=element_text(size=16))

library(pROC)
names(glmo$model.roc)
fit = #$finalModel
plot.roc(glmo$model.roc$pred$obs, glmo$model.roc$pred$Yes,col='red')
plot.roc(o$model.roc$pred$obs, o$model.roc$pred$Yes,add=TRUE,col='navy')


par(cex=1.2)
glm.roc = roc(glmo$model.roc$pred$obs, glmo$model.roc$pred$Yes)
rf.roc = roc(o$model.roc$pred$obs, o$model.roc$pred$Yes)
plot(glm.roc,col='red')
plot(rf.roc,add=TRUE,col='navy')

