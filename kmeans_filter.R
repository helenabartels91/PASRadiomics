reverse.map <- function(mm, form, data){
# Author: Ross Boylan {ross at biostat.ucsf.edu}
# At: https://stat.ethz.ch/pipermail/r-help/2013-July/357501.html
# return a vector v such that data[,v[i]] contributed to mm[,i]
# mm = model matrix produced by
# form = formula
# data = data
    tt <- terms(form, data=data)
    ttf <- attr(tt, "factors")
    mmi <- attr(mm, "assign")
    # this depends on assign using same order as columns of factors
    # entries in mmi that are 0 (the intercept) are silently dropped
    ttf2 <- ttf[,mmi]
    # take the first row that contributes
    r <- apply(ttf2, 2, function(is) rownames(ttf)[is > 0][1])
    match(r, colnames(data))
}

pca.kmeans.filter <- function(x,k=NULL,co=.95){
	require(NbClust)
	require(caret)
	form = as.formula("~.+0")
	xm = model.matrix(form,data=x)
	pca = prcomp(xm,scale.=TRUE)
	xs = pca$rotation	
	if(is.null(k)){ # need to identify k
		kpca = which(summary(pca)$importance[3,]>co)[[1]]
		k = min(kpca, ceiling(sqrt(ncol(xs))))
	}
	ko = kmeans(xs, centers=k)
	#
	# now find medoids
	medoids.m = numeric(k)
	for(ki in 1:k){
		k3 = which(ko$cluster==ki)
		if(length(k3)>1){
			med = apply(xs[k3,], 2, median)
			mM = matrix(med,ncol=length(med),nrow=length(k3),byrow=TRUE)
			ds = apply((xs[k3,]-mM)^2,1,mean)
			medoids.m[ki] = k3[which.min(ds)]
		} else {
			medoids.m[ki] = k3
		}
	}
	#
	# Now we need to translate this output in terms of 
	# the original feature frame
	map = reverse.map(xm,form,x) 
	clusters = numeric(ncol(x))
	for(p in 1:ncol(xm)){
		clusters[map[p]] = ko$cluster[p]
	}
	medoids = unique(map[medoids.m])
	medoid.names = names(x)[medoids]
	names(clusters) = names(x)
	#
	return(list(pca=pca,cluster=clusters,
			medoids=medoids,medoid.names=medoid.names,k=k))
}