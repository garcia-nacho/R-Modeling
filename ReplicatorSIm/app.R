#  Numerical simulation for a population of 
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
      sliderInput("Cell", "Number of Replicators:", 1,60,20),
      sliderInput("Pert", "Perturbation Strength:", 0,100,50),
      sliderInput("Pos", "Positive Interaction:", 0,100,50),
      sliderInput("Neg", "Negative Interaction:", 0,100,50),
      actionButton("do", "RUN", align = "center"),
      
      h6("By Nacho Garcia 2018", align = "left"),
      tags$a(href="https://github.com/garcia-nacho/R-Modeling/blob/master/ReplicatorSIm/app.R", "Source Code")
      
      
    ),
    
    # Plot
    mainPanel(
      
      tabsetPanel(
        tabPanel("Frequencies", plotOutput("Plot")),
        tabPanel("Count", plotOutput("Plot2")),
        tabPanel("Help", htmlOutput("help"))
      )
      
      
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
    R <- 3+R*-0.01
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
    total<-c(Cells)
    
    #Number of cells at t=0
    C[1,]<-1/Cells

    for (t in 2:tf){
      for(h in 1:Cells){
        #Include days of selection: High GR deselected
        
        C[t,h]<-(C[t-1,h]*exp((R[h]+FT)*t))
      }
      
      a<-sum(C[t,])
      total[t]<-a
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
    Abs<-as.data.frame(total)
    colnames(Abs)<-"Total"
    Abs$Time<-c(1:nrow(Abs))
    
    output$Plot <- renderPlot( { 

      q<-ggplot(C)+
        geom_line(aes(x=Time, y=Ratio, colour=variable),size=1.5)+
        theme_minimal(base_size = 20)+
        #ylim(0,1)+
        theme(legend.position="none")+
        ggtitle("Evolution of Ratios")
      
      print(q)
          })
    
    output$Plot2 <- renderPlot( { 
      
      r<-ggplot(Abs)+
        geom_line(aes(x=Time, y=log(Total)),size=1.5, colour="green")+
        theme_minimal(base_size = 20)+
        #ylim(0,1)+
        theme(legend.position="none")+
        ggtitle("Absolute Number")+
        ylab("log(Count)")
      
      print(r)
    })
    

  })        

  output$help <- renderUI({
    
    HTML(paste(
      "This numerical simulation provides an exploratory framework to study how positive or negative genetic perturbations evolve with time.",
      "",
      "<b>Number of Replicators:</b> Number of replicating entities to be simulated ",
      "",
      "<b>Perturbation Strength:</b> How much the fitness of the different replicators is affected by the perturbation (positive or negative)",
      "",    
      "<b>Positive interaction:</b> Strength of the positive interaction between replicators. This can be understood as replicators producing stimulating chemicals",
      "",
      "<b>Negative interaction:</b> Strength of the negative interaction between replicators. This can be understood as replicators producing toxic compounds",
      sep = "</br>"
      
      
    ))
    
  })
  
  }
# Run the application 
shinyApp(ui = ui, server = server)
