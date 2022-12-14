dat = read.csv('/Users/helen/Library/CloudStorage/OneDrive-Personal/Documents/Research/PAS Radiomics research/R code and files/Inferior_adherentinvasive_R.csv')
nms = names(dat)
dim(dat)

# target variable
y = as.factor(dat[,"Adherent_invasive"])
levels(y) = c("No","Yes")
dat$Adherent_invasive = NULL
table(y)

#set of predictor variables when #applied all variables included. If # removed from lines 12-28, these variables excluded (original image only)
#pps = c("log.sigma.2.0.mm.","log.sigma.3.0.mm.","log.sigma.4.0.mm.","log.sigma.5.0.mm.",
#		"wavelet.LLH_","wavelet.LHL_","wavelet.LHH_","wavelet.LHH_",
#		"wavelet.HLL_",
#		"wavelet.HLH_","wavelet.HHL_","wavelet.HHH_","wavelet.LLL_",
#		"exponential_","gradient_",
#		"lbp.3D.m1_","lbp.3D.m2_","lbp.3D.k_",
#		"logarithm","square_","squareroot_")
#ik = NULL
#for(i in 1:length(pps)){
#	ik = c(ik, grep(pps[i],names(dat)))
#}
#dat = dat[,-ik]
dim(dat)
nms = names(dat)

# ij = grep("Image",nms)[-c(7,9,length(ij)-c(0:2))]
# ij = c(ij, grep("Mask",nms)[-c(6,length(ij)-c(0:2))])
ij = c(grep("Study_ID",nms),
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
dat = dat[,-ij]
nms = names(dat)

# pre-filtering
# (correlation)
ic = nearZeroVar(dat)
if(length(ic)){ 
	dat = dat[,-ic]
}
M = cor(dat)
ic = findCorrelation(M, cutoff=0.8)
nms[ic]
dat.f = dat[,-ic]
dim(dat.f)
x = dat.f
