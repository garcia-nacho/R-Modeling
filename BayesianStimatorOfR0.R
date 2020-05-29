
df<-read.csv("/home/nacho/Covid19Dataset/Spain.csv")
data<-df$y1[c(12:65)]
cases <- df$ConfirmedCases[c(12:65)]
#Limits [-0.3, 3]

# limits [0, 2] R0<-R0/2
data<-data/2
data

# True distribution
p <- 0.3
n <- 1

# Prior on p (n assumed known), discretized
p_values <- seq(0.01,0.99,0.001)

pr <- dbeta(p_values,1,1)
pr <- 1000 * pr / sum(pr)  # Have to normalize given discreteness
plot(pr~p_values, col=1, ylim=c(0, 14), type="l")

sampled<-vector()
predicted<-vector()
for (i in 1:length(data)) {
  
  x <- rbinom(1, n, data[i]) 
  ps <- dbinom(x, n, p_values) * pr
  ps <- 1000 * ps / sum(ps)
  
  predicted<-c(predicted,p_values[which(ps==max(ps))])
  
  #Sampling
  binom <- rbinom(cases[i], length(p_values), predicted[i])
  sampled<-c(sampled, p_values[round(mean(binom))])
  
  lines(ps~p_values, col=(i+1))
  
  pr = ps
}

plot(data, type = "l")
lines(predicted, col="red")

cases.pred<-rep(1,length(cases))
cases.samp<-rep(1,length(cases))
predicted<-predicted*2
sampled<-sampled*2

for (i in 2:length(cases)) {
  cases.pred[i]<-cases[i-1]+ (cases[i-1]*predicted[i])
  cases.samp[i]<-cases[i-1]+ (cases[i-1]*sampled[i])
}
plot(cases.pred, type = "l")
lines(cases, col="red")
lines(cases.samp, col="blue")
