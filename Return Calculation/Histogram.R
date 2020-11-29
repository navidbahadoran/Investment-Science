library("moments")
rm(list = ls(all.names = TRUE))
cat("\f")
# Reading csv files
MSFTpath<-file.path("C:/Users/Azar/Documents/R_Examples", "MSFT_d.csv")
AMZNpath<-file.path("C:/Users/Azar/Documents/R_Examples", "AMZN_d.csv")
GOOGpath<-file.path("C:/Users/Azar/Documents/R_Examples", "GOOG_d.csv")

MSFTPc <- read.csv(MSFTpath)
AMZNPc <- read.csv(AMZNpath)
GOOGPc<-read.csv(GOOGpath)

# Separating Adjusted close price for each stock
MSFT.AdjClose <- MSFTPc[,"Adj.Close"] 
AMZN.AdjClose <- AMZNPc[,"Adj.Close"] 
GOOG.AdjClose <- GOOGPc[,"Adj.Close"] 

# calculate Arithmetic return
MSFTRtn<-(MSFT.AdjClose[-1]/MSFT.AdjClose[-length(MSFT.AdjClose)])-1
AMZNRtn<-(AMZN.AdjClose[-1]/AMZN.AdjClose[-length(AMZN.AdjClose)])-1
GOOGRtn<-(GOOG.AdjClose[-1]/GOOG.AdjClose[-length(GOOG.AdjClose)])-1

# Microsoft Mean, Variance, Skewness and Kurtosis and its histogram vs normal dist.

MSFTlength<-length(MSFTRtn)
MeanMSFT<-mean(MSFTRtn)
cat("Microstoft Return Mean:",MeanMSFT,"\n")
VarMSFT<-var(MSFTRtn)*(MSFTlength-1)/MSFTlength   #calculation of population variance through sample variance
cat("Microstoft Return Variance:",VarMSFT,"\n")
SkewMSFT<-skewness(MSFTRtn)
cat("Microsoft Return Skewness:",SkewMSFT,"\n")
KurMSFT<-kurtosis(MSFTRtn)
cat("Microstoft Return Kurtosis:",KurMSFT,"\n")
MinMSFT<-min(MSFTRtn)
MaxMSFT<-max(MSFTRtn)
MSFTx<-seq(MinMSFT,MaxMSFT,length=1000)
MSFTy<-dnorm(MSFTx,mean = MeanMSFT,sd=sqrt(VarMSFT))

# Amazon Mean, Variance, Skewness and Kurtosis and its histogram vs normal dist.

AMZNlength<-length(AMZNRtn)
MeanAMZN<-mean(AMZNRtn)
cat("Amazon Return Mean:",MeanAMZN,"\n")
VarAMZN<-var(AMZNRtn)*(AMZNlength-1)/(AMZNlength)   #calculation of population variance through sample variance
cat("Amazon Return Variance:",VarAMZN,"\n")
SkewAMZN<-skewness(AMZNRtn)
cat("Amazon Return Skeness:",SkewAMZN,"\n")
KurAMZN<-kurtosis(AMZNRtn)
cat("Amazon Return Kurtosis:",KurAMZN,"\n")
MinAMZN<-min(AMZNRtn)
MaxAMZN<-max(AMZNRtn)
AMZNx<-seq(MinAMZN,MaxAMZN,length=1000)
AMZNy<-dnorm(AMZNx,mean = MeanAMZN,sd=sqrt(VarAMZN))


# Google Mean, Variance, Skewness and Kurtosis and its histogram vs normal dist.

GOOGlength<-length(GOOGRtn)
MeanGOOG<-mean(GOOGRtn)
cat("Google Return Mean:",MeanGOOG,"\n")
VarGOOG<-var(GOOGRtn)*(GOOGlength-1)/GOOGlength   #calculation of population variance through sample variance
cat("Google Return Variance:",VarGOOG,"\n")
SkewGOOG<-skewness(GOOGRtn)
cat("Google Return Skewness:",SkewGOOG,"\n")
KurGOOG<-kurtosis(GOOGRtn)
cat("Google Return Kurtosis:",KurGOOG,"\n")
MinGOOG<-min(GOOGRtn)
MaxGOOG<-max(GOOGRtn)
GOOGx<-seq(MinGOOG,MaxGOOG,length=1000)
GOOGy<-dnorm(GOOGx,mean = MeanGOOG,sd=sqrt(VarGOOG))


#plotting the histograms and normal pdf

par(mfrow=c(3,1))
dev.new()
# plot Microsoft
hist(MSFTRtn,breaks = 40,ylim = c(0,50),freq = FALSE,xlab = "return",ylab = "f(x)",
     main = "Histogram of Microsoft Return",xaxt='n')
axis(side = 1,at=seq(round(MinMSFT,2),round(MaxMSFT+0.01,2),0.1))
lines(MSFTx,MSFTy,col="red",lwd=2)
dev.new()
# plot AMazon
hist(AMZNRtn,breaks = 40,ylim = c(0,50), freq = FALSE,xlab = "return",ylab = "f(x)",
     main = "Histogram of Amazon Return",xaxt='n')
axis(side=1, at=seq(round(MinAMZN,2),round(MaxAMZN+0.04,2), 0.04))
lines(AMZNx,AMZNy,col="red",lwd=2)
dev.new()
# plot Google
hist(GOOGRtn,breaks = 40,ylim = c(0,50),xlim = c(MinGOOG-0.005,MaxGOOG+0.005), freq = FALSE,xlab = "return",ylab = "f(x)",
     main = "Histogram of Google Return",xaxt='n')
axis(side=1, at=seq(round(MinGOOG-0.01,2),round(MaxGOOG+0.04,2), 0.03))
lines(GOOGx,GOOGy,col="red",lwd=2)