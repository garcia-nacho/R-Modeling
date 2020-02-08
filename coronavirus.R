library(gapminder)
library(gganimate)
library(ggplot2)

# Alive 1
# Dead 0
# Infected 3 
# Immune 2

#Parameters
time<-100
Infection_rate<-0.01
Pop_density<- 10
Total_pop<-10000
Survival_rate<-0.9777
silent_infection<-9
contact_rate<-40

df<-rep(1, Total_pop)

patient_zero<-sample(which(df==1),1)
df[patient_zero]<-3

vaccinated<-sample(which(df==1),round(Total_pop*0.6))
df[vaccinated]<-2

progression<-matrix(data = NA, nrow =  time, ncol=Total_pop)

for(i in 1:time){
  
  #new
  # if(i>15 & length(which(df==1))>=200) {
  #       df[sample(which(df==1),200)]<-2
  # }
  
  #if(i>15) contact_rate<-10
  
  df[df>=3]<-df[df>=3]+1
  #Infections
  if(length(df>=3)>0){
  for (k in 1:length(which(df>=3))) {
    round_contacts<-sample(which(df!=0),contact_rate)
    new_infected<-vector()
    
    for (l in 1:length(round_contacts)) {
      if(runif(1, min = 0, max = 1)<=Infection_rate & df[round_contacts[l]]==1) new_infected<-c(new_infected,round_contacts[l] )
      
    }
    new_infected<-length(new_infected)
    if(new_infected>0){
      
      if(length(which(df==1))<new_infected){
        new_infected<-length(which(df==1))
      }
    
    new_infected<-sample(which(df==1),new_infected)
    df[new_infected]<-3}
  }
  
  #Deaths
  infected<-which(df>=3)
  for (j in 1:length(infected)) {
    if(runif(1, min=0, max = 1)< ((1-Survival_rate)/silent_infection)) df[infected[j]]<-0
  }}
  if(length(which(df>=3+silent_infection))>0) df[which(df>=3+silent_infection)]<-2
  progression[i,]<-df
}



df.plot<-expand.grid(c(1:100),c(1:100))
colnames(df.plot)<-c("X","Y")
df.plot$Time<-1
df.plot$Data<-progression[1,]

  for (i in 1:nrow(progression)) {
  dummy<-df.plot
  dummy$Time<-i
  dummy$Data<-progression[i,]
  if(!exists("to.plot")){
    to.plot<-dummy
  }else{
    to.plot<-rbind(to.plot,dummy)
  }
  }

to.plot$Data[to.plot$Data>=3]<-3

ggplot(to.plot, aes(X, Y,  fill = as.factor(Data))) +
  geom_tile()+
  transition_time(time = Time) +
  scale_fill_manual(name="Status",
                     labels=c("Dead","Susceptible","Immune","Infected"),
                     values=c("red", "black","blue","green"))+

  labs(title = 'Day: {frame_time}') +
  xlim(0,100)+
  ylim(0,100)+
  ease_aes('linear')+
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())+
  theme_void()


####Inference
df<-read.csv("/home/nacho/InfectedCV.csv")

ggplot(df)+
  geom_point(aes(Day, Infected), size=3, colour="red")+
  geom_line(aes(Day, Infected))+
  theme_minimal()

y<-df$Infected
x<-df$Day

linear<-lm(log(y-1)~x)
start <- list(a=exp(coef(linear)[1]), b=coef(linear)[2])
mod <- nls(Infected ~ a*(exp(b*Day)), data = df, start = start)

a<-summary(mod)$coefficients[1,1]
b<-summary(mod)$coefficients[2,1]

Day<-75
prediction <- a*(exp(b*Day))

a*(exp(b*101))/(a*(exp(b*100)))

df$predicted<- a*(exp(b*df$Day))

ggplot(df)+
  geom_point(aes(Day, Infected), size=3, colour="red")+
  geom_point(aes(Day[13], Infected[13]), size=3, colour="blue")+
  #geom_line(aes(Day, Infected))+
  geom_line(aes(Day, predicted), colour="blue")+
  theme_minimal()

(log(7627710110)- log(a))/b
