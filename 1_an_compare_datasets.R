# Eric Wolsztynski, University College Cork, 2023
# eric.w@ucc.ie

rm(list=ls())

require(caret)
require(pROC)
source("src_validation.r")

# load the data
# check which Irish dataset is the correct version:
datIinf = read.csv('XX.csv')
datIsup = read.csv('XX.csv')
#datIinf$BMI = NULL
#datIsup$BMI = NULL
datLinf = read.csv('XX.csv')
datLsup = read.csv('XX.csv')
# make sure features match throughout
dim(datIinf); dim(datIsup) 
dim(datLinf); dim(datLsup) 
# fix Study.ID naming in datLinf
names(datLinf)[1] = names(datLsup)[1]
# fix MRI_scanner in datLinf
names(datLinf)[3] = names(datIinf)[3]
names(datLsup)[3] = names(datIsup)[3]
# reorder datLinf
datLinf = datLinf[,names(datLsup)]
# final check
identical(names(datLsup),names(datLinf))
identical(names(datIinf),names(datIsup))
identical(names(datLinf),names(datLsup))
identical(sort(names(datIsup)),sort(names(datLsup)))
identical(sort(names(datIinf)),sort(names(datLinf)))
# reorder datLinf and datLsup
datLinf = datLinf[,names(datIinf)]
datLsup = datLsup[,names(datIsup)]
# check for NA's
sum(is.na(datIinf)); sum(is.na(datIsup))
sum(is.na(datLinf)); sum(is.na(datLsup))

save.image(file="out/pooled_data.RData")

# no reason to believe the Y-distributions differ:
tbI = table(datIinf$Adherent_invasive)
tbL = table(datLinf$Adherent_invasive)
chisq.test(cbind(tbI,tbL)) 

xIinf = datIinf[,-c(1:3)]
xIsup = datIsup[,-c(1:3)]
xLinf = datLinf[,-c(1:3)]
xLsup = datLsup[,-c(1:3)]

# check for NZV
nzIinf = nearZeroVar(xIinf)
nzIsup = nearZeroVar(xIsup)
nzLinf = nearZeroVar(xLinf)
nzLsup = nearZeroVar(xLsup)
names(xIinf)[nzIinf]
names(xIsup)[nzIsup]
names(xLinf)[nzLinf]
names(xLsup)[nzLsup]
# more NZV features in Isup than Iinf:
setdiff(nzIinf,nzIsup)
setdiff(nzIsup,nzIinf) 
ic = setdiff(nzIinf,nzIsup); names(xIinf)[ic]
# more NZV features in Lsup than Linf:
setdiff(nzLinf,nzLsup)
setdiff(nzLsup,nzLinf)
ic = setdiff(nzLinf,nzLsup); names(xLinf)[ic]
# more NZV features in Linf than Iinf :
setdiff(nzIinf,nzLinf)
setdiff(nzLinf,nzIinf) 
ic = setdiff(nzLinf,nzIinf); names(xLinf)[ic]
# more NZV features in Lsup than Isup:
setdiff(nzIsup,nzLsup)
setdiff(nzLsup,nzIsup) 
ic = setdiff(nzLsup,nzIsup); names(xLsup)[ic]
# --> decision to remove all lbp.3D features from subsequent analyses
i.lbp.3D = grep("lbp.3D.",names(xIinf))
xIinf = xIinf[,-i.lbp.3D]
xIsup = xIsup[,-i.lbp.3D]
xLinf = xLinf[,-i.lbp.3D]
xLsup = xLsup[,-i.lbp.3D]

COLS_IL = c('lightgreen','tomato')
o.Iinf.Linf = compare.datasets(xIinf,xLinf,fnm="Iinf_vs_Linf.pdf",
		COLS=COLS_IL,LABS=c('Ireland_inf','Lebanon_inf'))
# List of significantly different features:
is1 = which(o.Iinf.Linf$p.values[,1]<0.05)
paste("There are",length(is1),"different features, out of",
	nrow(o.Iinf.Linf$p.values))
# o.Iinf.Linf$p.values[is1,]
o.Isup.Lsup = compare.datasets(xIsup,xLsup,fnm="Isup_vs_Lsup.pdf",
		COLS=COLS_IL,LABS=c('Ireland_sup','Lebanon_sup'))
# List of significantly different features:
is2 = which(o.Isup.Lsup$p.values[,1]<0.05)
paste("There are",length(is2),"different features, out of",
	nrow(o.Isup.Lsup$p.values))
# o.Isup.Lsup$p.values[is2,]
length(intersect(names(o.Iinf.Linf$p.values[is1,]),names(o.Isup.Lsup$p.values[is2,])))
o.Iinf.Linf$p.values[1:2,]
# setdiff(names(o.Iinf.Linf$p.values[is1,]),names(o.Isup.Lsup$p.values[is2,]))
# setdiff(names(o.Isup.Lsup$p.values[is2,]),names(o.Iinf.Linf$p.values[is1,]))

# features with sig. different distributions:
ncol(xIinf)
ncol(xLinf)
ncol(xIsup)
ncol(xLsup)
ds1 = which(o.Iinf.Linf$p.values[,1]<0.05)
ds2 = which(o.Isup.Lsup$p.values[,1]<0.05)
length(ds1)
length(ds2)
length(intersect(ds1,ds2))

# subset of features with no sig. different distributions:
js1 = which(o.Iinf.Linf$p.values[,1]>=0.05)
js2 = which(o.Isup.Lsup$p.values[,1]>=0.05)
icset = intersect(js1,js2)
# use either whole set (comment out whichever one is not desired):
cset = names(xIinf)
# or intersection subset:
# cset = names(xIinf)[icset]
length(icset)
cset

# UV analyses (on common feature set)
uvmw.Iinf = uv.mw(x=xIinf[,cset],y=datIinf$Adherent_invasive)
uvmw.Isup = uv.mw(x=xIsup[,cset],y=datIsup$Adherent_invasive)
uvmw.Linf = uv.mw(x=xLinf[,cset],y=datLinf$Adherent_invasive)
uvmw.Lsup = uv.mw(x=xLsup[,cset],y=datLsup$Adherent_invasive)
sum(uvmw.Iinf<0.05)
sum(uvmw.Isup<0.05)
sum(uvmw.Linf<0.05)
sum(uvmw.Lsup<0.05)
uvmw.Linf[which(uvmw.Linf<0.05)]

mod <- function(x){
	x = as.factor(x)
	if(levels(x)[1]!="0"){ stop() }
	levels(x) = c("No","Yes")
	return(x)
}

datIinf$Adherent_invasive = mod(datIinf$Adherent_invasive)
datLinf$Adherent_invasive = mod(datLinf$Adherent_invasive)
datIsup$Adherent_invasive = mod(datIsup$Adherent_invasive)
datLsup$Adherent_invasive = mod(datLsup$Adherent_invasive)
uva.Iinf = uv.analyses(x=xIinf[,cset],y=datIinf$Adherent_invasive)
uva.Isup = uv.analyses(x=xIsup[,cset],y=datIsup$Adherent_invasive)
uva.Linf = uv.analyses(x=xLinf[,cset],y=datLinf$Adherent_invasive)
uva.Lsup = uv.analyses(x=xLsup[,cset],y=datLsup$Adherent_invasive)
#
sens.Iinf = apply(uva.Iinf$sens,2,mean); max(sens.Iinf)
spec.Iinf = apply(uva.Iinf$spec,2,mean); max(spec.Iinf)
aucs.Iinf = apply(uva.Iinf$aucs,2,mean); max(aucs.Iinf)
accs.Iinf = apply(uva.Iinf$accs,2,mean); max(accs.Iinf)
#
sens.Isup = apply(uva.Isup$sens,2,mean); max(sens.Isup)
spec.Isup = apply(uva.Isup$spec,2,mean); max(spec.Isup)
aucs.Isup = apply(uva.Isup$aucs,2,mean); max(aucs.Isup)
accs.Isup = apply(uva.Isup$accs,2,mean); max(accs.Isup)
#
sens.Linf = apply(uva.Linf$sens,2,mean); max(sens.Linf)
spec.Linf = apply(uva.Linf$spec,2,mean); max(spec.Linf)
aucs.Linf = apply(uva.Linf$aucs,2,mean); max(aucs.Linf)
accs.Linf = apply(uva.Linf$accs,2,mean); max(accs.Linf)
#
sens.Lsup = apply(uva.Lsup$sens,2,mean); max(sens.Lsup)
spec.Lsup = apply(uva.Lsup$spec,2,mean); max(spec.Lsup)
aucs.Lsup = apply(uva.Lsup$aucs,2,mean); max(aucs.Lsup)
accs.Lsup = apply(uva.Lsup$accs,2,mean); max(accs.Lsup)

combine <- function(fIinf,fIsup,fLinf,fLsup,lab="AUC"){
	vals = cbind(fIinf,fIsup,fLinf,fLsup)
	nms = c("Irish.inf","Irish.sup","Leban.inf","Leban.sup")
	boxplot(vals,col=c('cyan','pink'),ylim=c(0,1),main=lab,names=nms)
	p.I = wilcox.test(fIinf,fIsup)$p.value
	p.L = wilcox.test(fLinf,fLsup)$p.value
	p.inf = wilcox.test(fIinf,fLinf)$p.value
	p.sup = wilcox.test(fIsup,fLsup)$p.value
	return(c(p.I,p.L,p.inf,p.sup))
}

par(mfrow=c(2,2),mar=c(4,3,3,1),pch=20)
p.aucs = combine(aucs.Iinf, aucs.Isup, aucs.Linf, aucs.Lsup, lab="AUC")
p.accs = combine(accs.Iinf, accs.Isup, accs.Linf, accs.Lsup, lab="Accuracy")
p.sens = combine(sens.Iinf, sens.Isup, sens.Linf, sens.Lsup, lab="Sensitivity")
p.spec = combine(spec.Iinf, spec.Isup, spec.Linf, spec.Lsup, lab="Specificity")

pv = rbind(p.aucs,p.accs,p.sens,p.spec)
rownames(pv) = c("AUC","Acc","Sens","Spec")
colnames(pv) = c("Irish inf v sup","Leban. inf v sup",
				"Irish inf v Leban. inf", "Irish sup v Leban. sup")
round(t(pv),4)

# pooled comparisons (combining Irish+Lebanese)
pool.inf = rbind(datIinf,datLinf)
pool.sup = rbind(datIsup,datLsup)
x.inf = rbind(xIinf,xLinf)
x.sup = rbind(xIsup,xLsup)

# poooled univariate analyses...
# ... on combined feature subset
uvmw.pooled.inf = uv.mw(x=pool.inf[,cset],y=pool.inf$Adherent_invasive)
uvmw.pooled.sup = uv.mw(x=pool.sup[,cset],y=pool.sup$Adherent_invasive)
uva.pooled.inf = uv.analyses(x=xIinf[,cset],y=datIinf$Adherent_invasive)
uva.pooled.sup = uv.analyses(x=xIsup[,cset],y=datIsup$Adherent_invasive)
min(uvmw.pooled.inf)
mean(uvmw.pooled.sup<0.05); uvmw.pooled.sup[which(uvmw.pooled.sup<0.05)]
dim(uva.pooled.inf$aucs)
sens.inf = apply(uva.pooled.inf$sens,2,mean); max(sens.inf)
spec.inf = apply(uva.pooled.inf$spec,2,mean); max(spec.inf)
aucs.inf = apply(uva.pooled.inf$aucs,2,mean); max(aucs.inf)
accs.inf = apply(uva.pooled.inf$accs,2,mean); max(accs.inf)
sens.sup = apply(uva.pooled.sup$sens,2,mean); max(sens.sup)
spec.sup = apply(uva.pooled.sup$spec,2,mean); max(spec.sup)
aucs.sup = apply(uva.pooled.sup$aucs,2,mean); max(aucs.sup)
accs.sup = apply(uva.pooled.sup$accs,2,mean); max(accs.sup)


# ... on whole feature set
uvmw.pooled.inf.w = uv.mw(x=pool.inf[,-c(1:3)],y=pool.inf$Adherent_invasive)
uvmw.pooled.sup.w = uv.mw(x=pool.sup[,-c(1:3)],y=pool.sup$Adherent_invasive)
uva.pooled.inf.w = uv.analyses(x=xIinf,y=datIinf$Adherent_invasive)
uva.pooled.sup.w = uv.analyses(x=xIsup,y=datIsup$Adherent_invasive)
min(uvmw.pooled.inf.w)
mean(uvmw.pooled.sup.w<0.05); uvmw.pooled.sup.w[which(uvmw.pooled.sup.w<0.05)]
sens.inf = apply(uva.pooled.inf.w$sens,2,mean); max(sens.inf)
spec.inf = apply(uva.pooled.inf.w$spec,2,mean); max(spec.inf)
aucs.inf = apply(uva.pooled.inf.w$aucs,2,mean); max(aucs.inf)
accs.inf = apply(uva.pooled.inf.w$accs,2,mean); max(accs.inf)
sens.sup = apply(uva.pooled.sup.w$sens,2,mean); max(sens.sup)
spec.sup = apply(uva.pooled.sup.w$spec,2,mean); max(spec.sup)
aucs.sup = apply(uva.pooled.sup.w$aucs,2,mean); max(aucs.sup)
accs.sup = apply(uva.pooled.sup.w$accs,2,mean); max(accs.sup)
#
which.max(sens.inf)
which.max(spec.inf)
which.max(aucs.inf)
which.max(accs.inf)
which.max(sens.sup)
which.max(spec.sup)
which.max(aucs.sup)
which.max(accs.sup)

save.image(file="out/univariate_analyses.Rdata")

fun <- function(kpi){
	means = apply(kpi,2,mean)
	imax = which.max(means)
	nm = colnames(kpi)[imax]
	mkpi = kpi[,imax]
	if(0){
		cat(nm,round(means[imax],3),
			", 2.5%=",round(quantile(mkpi,0.025),3),
			", 97.5%=",round(quantile(mkpi,0.975),))
	}
	return(data.frame(name=nm,mean=round(means[imax],3),
			P0.025=round(quantile(mkpi,0.025),3),
			P0.975=round(quantile(mkpi,0.975),3)))
}

df.inf = rbind(fun(uva.pooled.inf.w$aucs),
fun(uva.pooled.inf.w$sens),
fun(uva.pooled.inf.w$spec),
fun(uva.pooled.inf.w$accs))
rownames(df.inf) = c("AUC","Sens","Spec","Acc")
df.inf

df.sup = rbind(fun(uva.pooled.sup.w$aucs),
fun(uva.pooled.sup.w$sens),
fun(uva.pooled.sup.w$spec),
fun(uva.pooled.sup.w$accs))
rownames(df.sup) = c("AUC","Sens","Spec","Acc")
df.sup

fun.box <- function(kpi){
	means = apply(kpi$aucs,2,mean)
	box = cbind(kpi$aucs[,which.max(means)])
	means = apply(kpi$sens,2,mean)
	box = cbind(box, kpi$sens[,which.max(means)])
	means = apply(kpi$spec,2,mean)
	box = cbind(box, kpi$spec[,which.max(means)])
	means = apply(kpi$accs,2,mean)
	box = cbind(box, kpi$accs[,which.max(means)])
	colnames(box) = c("AUC","Sens","Spec","Acc") 
	return(box)
}
par(mfrow=c(2,1),font=2,pch=20,mar=c(4,3,2,1))
obox = fun.box(uva.pooled.inf.w)
boxplot(obox,main="Inferior region")
obox = fun.box(uva.pooled.sup.w)
boxplot(obox,main="Superior region")

# pooled comparisons (combining Irish+Lebanese)
pool.inf = rbind(datIinf,datLinf)
pool.sup = rbind(datIsup,datLsup)

save.image(file="out/pooled_data.RData")
write.csv(pool.inf, file = "pooledinferior.csv", row.names = FALSE)
write.csv(pool.sup, file = "pooledsuperior.csv", row.names = FALSE)
idx = order(spec.inf,decreasing=TRUE)[1:10]
nm = names(spec.inf[idx])
fm = spec.inf[idx]
cbind(fm) # to display the top 10 performance
M = cor(pool.inf[,nm])
diag(M) = 0
MX = round(M,4)
colnames(MX) = c(1:10)
MX # to display the correlation matrix for this top 10
