# Eric Wolsztynski, University College Cork, 2023
# eric.w@ucc.ie

rm(list=ls())
load(file="out/pooled_data.RData")
source("src_validation.r")
# source('update.R')
library(caret)
library(MASS)

# remove patient info
x.inf = pool.inf[,-c(1:2)]
x.sup = pool.sup[,-c(1:2)]

y.inf = as.factor(pool.inf$Adherent_invasive)
y.sup = as.factor(pool.sup$Adherent_invasive)
levels(y.inf) = c("No","Yes")
levels(y.sup) = c("No","Yes")
y.inf = relevel(y.inf,ref="Yes")
y.sup = relevel(y.sup,ref="Yes")

x = x.inf
y = y.inf

nms = names(x)
# filter names
ftypes = nms[grep("glrlm_GrayLevelVariance",nms)]
ftypes = unlist(strsplit(ftypes,split="_glrlm_GrayLevelVariance"))
# h-o features
ho.set = c(grep("glrlm",nms))
ho.set = c(ho.set,grep("gldmm",nms))
ho.set = c(ho.set,grep("glszm",nms))
# l-o features
lo.set = c(3:length(nms))[-ho.set]
#
# sset = ho.set
# sset = lo.set
sset = c(lo.set,ho.set)
sset = c(1:nrow(x))

SEEDS = c(1219,1912,4060,4061,6040)
SEED = SEEDS[3]
set.seed(SEED)
k = 3
B = 30
tG.rr = expand.grid(alpha = 0, lambda = (1:30) * .01)
tG.rf = expand.grid(mtry = seq(4,6,by=1))

R = 5000
MSIZE = 5
oz = NULL
sels = matrix(0,nrow=R,ncol=length(sset))
colnames(sels) = nms[sset]
res.ho = matrix(NA,nrow=R,ncol=4)
for(i in 1:R){
	is = nms[sample(sset,size=MSIZE)]
	sels[i,is] = 1
	# is = c("No.of.CS",is)
	xm = x[,is]	
	o = train.class(x=xm,y=y,model='rf',k=k,B=B,
		method="repeatedcv",tG=tG.rf)
	oz[[i]] = o
	o.acc = mean(o$model.acc$resample$Accuracy)
	o.sens = mean(o$model.roc$resample$Sens)
	o.spec = mean(o$model.roc$resample$Spec)
	o.auc = mean(o$model.roc$resample$ROC)
	out = cbind(o.auc,o.acc,o.sens,o.spec)
	res.ho[i,] = out
	if(i%%100==0){
		cat("Loop...",i,"\n")
		save.image(file=paste("out/run_",R,"_hilo_",SEED,"_R",B,"k",k,"_MSIZE_",MSIZE,".Rdata",sep=""))
	}
}
colnames(res.ho) = c("auc","acc","sens","spec")
rk = order(res.ho[,"sens"],decreasing=TRUE)
res.ho[rk[1:10],]

TARGET = 1
opt.set = nms[sset[which(sels[rk[1],]==TARGET)]]
opt.set
xf = cor.filter(nzv.filter(x),cutoff=.95)
intersect(opt.set,names(xf))
xf = nzv.filter(x)
intersect(opt.set,names(xf))
M = cor(x[,opt.set])
rownames(M) = colnames(M) = paste("x",1:ncol(M),sep='')
round(M,3)
diag(M) = 0
max(abs(M))
mean(res.ho[,'auc']>.6) # about 20% again
mean(res.ho[,'auc']>.7) # about 1% 

# if(0){
	# opt.set
	# xm = x[,opt.set]	
	# o = train.class(x=xm,y=y,model='rf',
		# method="repeatedcv",tG=tG.rf)
# } else {
	# o = oz[[rk[1]]]
# }
# mean(o$model.acc$resample$Accuracy)
# mean(o$model.roc$resample$Sens)
# mean(o$model.roc$resample$Spec)
# mean(o$model.roc$resample$ROC)
# probs = seq(.4, 0.6, by = 0.02) 
# ths <- thresholder(o$model.roc,
                   # threshold = probs,
                   # final = TRUE,
                   # statistics = "all")
# ths

# if(1){ # re-run and glm
	# om = opt.set
	# xm = x[,om]	
	# set.seed(1219)
	# ro = rfe(x=xm,y=y,sizes=c(3:6),metric="Sens",
			# rfeControl=rfeControl(functions=rfFuncs,
							# repeats=30,number=3,method="repeatedcv"))
	# ro
	
	# # try a glmnet on this same model?
	# xo = x[,opt.set]
	# tG.rr = expand.grid(alpha = 0, lambda = (1:30) * .01)
	# oo = train.class(x=xo,y=y,model='glmnet',
		# method="repeatedcv",tG=tG.rr)
	# oo.acc = mean(oo$model.acc$resample$Accuracy)
	# oo.sens = mean(oo$model.roc$resample$Sens)
	# oo.spec = mean(oo$model.roc$resample$Spec)
	# oo.auc = mean(oo$model.roc$resample$ROC)
	# outoo = cbind(oo.auc,oo.acc,oo.sens,oo.spec)
	# outoo
# }

