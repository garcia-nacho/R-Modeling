#  Numerical simulation for eplicators interacting
#  Nacho garcia 2018
#  garcia.nacho ---AT--- gmail.com
#  GPL v3

#Library loading
library(shiny)
library(ggplot2)
library(reshape)


# Define UI
ui <- fluidPage(
  
  # Application title
  titlePanel("Population Simulator"),
  
  # Sidebar 
  sidebarLayout(
    sidebarPanel(
      sliderInput("Cell", "Number of replicators:", 1,60,20),
      sliderInput("Pert", "Perturbation strength:", 0,100,50),
      sliderInput("Pos", "Positive interaction:", 0,100,50),
      sliderInput("Neg", "Negative interaction:", 0,100,50),
      actionButton("do", "RUN", align = "center"),
      
      h6("By Nacho Garcia 2018", align = "left"),
      tags$a(href="https://github.com/garcia-nacho/R-Modeling/blob/master/ReplicatorSIm/app.R", "Source Code")
      
      
    ),
    
    # Plot
    mainPanel(
      
      plotOutput("Plot")
      
    )
  )
)

# Define server 
server <- function(input, output) {
  
  observeEvent(input$do, {
    
    #Number of cells
    Cells<-input$Cell
    Pert<-(input$Pert+1)/100  #Range (0.1-1)
    R <-rlnorm(Cells,1,Pert)
    R <- 3+R*-0.1
    tf<-20
    
    FTc<-c(1:tf)
    FTc[1]<-0
    
    #Positive and negative interactors
    P<- runif(Cells, min =0, max = input$Pos/10)
    N<- runif(Cells, min =0, max = input$Neg/10)
    
    #Matrix Creation
    C<-as.data.frame(matrix(0,nrow = tf, ncol =Cells ))
    
    #Interaction 0 at t=0
    FT<-0
    
    #Number of cells at t=0
    C[1,]<-1/Cells

    for (t in 2:tf){
      for(h in 1:Cells){
        #Include days of selection: High GR deselected
        
        C[t,h]<-(C[t-1,h]*exp((R[h]+FT)*t))
      }
      
      a<-sum(C[t,])
      for(v in 1:Cells){
        
        C[t,v]<-C[t,v]/a
      }
      
      for (i in 1:Cells){
        FT<-FT + C[t,i]*P[i]-C[t,i]*N[i]
      }
      Ncel<-sum(C[t,])
      
      FTc[t]<-FT

    }
    
    
    C$t<-c(1:tf)
    C<-melt(C, id.vars = "t")
    colnames(C)<-c("Time","variable","Ratio")
    
    output$Plot <- renderPlot( { 

      q<-ggplot(C)+
        geom_line(aes(x=Time, y=Ratio, colour=variable))+
        theme_minimal()+
        ylim(0,1)+
        theme(legend.position="none")
      
      print(q)
      
      
      
    })
    
  })        
}
# Run the application 
shinyApp(ui = ui, server = server)

