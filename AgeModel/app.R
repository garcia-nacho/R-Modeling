
#

library(shiny)

# Define UI for application that draws a histogram
ui <- fluidPage(
   
   # Application title
   titlePanel("Population dynamics"),
   
   # Sidebar with a slider input for number of bins 
   sidebarLayout(
      sidebarPanel(
        actionButton("do", "Simulate", align = "center"),
        h6(" "),
        sliderInput("NutrientsSize", "Total Food Supply:", 10000,500000,100000),
        sliderInput("Food2Divide", "Food Required to replicate :", 5,50,20),
        sliderInput("FoodIntake", "Food intake/time unit", 1,50,10),
        sliderInput("DieEvent", "Programmed death at age:", 0,30,20),
        sliderInput("iterations", "Iterations:", 10,100,40),
        sliderInput("Reserve", "Internal Energy:", 1,50,10),
        sliderInput("SenFac", "Senescence Factor:", 1,100,50),
        
        
        
        h6("By Nacho Garcia 2018", align = "left"),
        tags$a(href="https://github.com/garcia-nacho/R-Modeling/blob/master/ElectricalFilter/app.R", "Source Code")
        
      ),
      
      # Show a plot of the generated distribution
      mainPanel(
        tabsetPanel(
          tabPanel("Plots", plotOutput("Plot"), plotOutput("Plot2"), plotOutput("Plot3")),
          tabPanel("Summary", tableOutput("summary")),
          tabPanel("Help", htmlOutput("help"))
        )
        
      )
   )
)

# Define server logic 
server <- function(input, output) {
   
   
  observeEvent(input$do, {
   
    NutrientsSize <- input$NutrientsSize
    Food2Divide <-input$Food2Divide
    FoodIntake <- input$FoodIntake
    DieEvent<-input$DieEvent
    iterations<-input$iterations
    Reserve<-input$Reserve
    IE<-input$Reserve
    Senesc<-input$SenFac
   
    
    ####
    
    Culture<-c(1)
    Reserve<-c(10)
    Food<- rep(1, NutrientsSize)
    Summary<-as.data.frame(c(1:iterations))
    colnames(Summary)<-c("Cells")
    
    Summary$Cells<-0
    Summary$Food<-0
    Summary$Time<-0
    Summary$AvrAge<-0
    Summary$EntropyAge<-0
    Summary$EntropyIE<-0
    

    
    for (i in 1:iterations){
      
      for (h in 1:length(Culture))
      {
        Meal<-sample(NutrientsSize,FoodIntake,replace = FALSE)
        Intake<-sum(Food[Meal])
        Food[Meal]<-0
        if(is.na(Reserve[h]))
        {
          Reserve[h]<-10
        }
        Reserve[h]<-Reserve[h]+Intake
        Reserve[h]<-Reserve[h]-as.integer(Culture[h]*Senesc/50)
        if(Reserve[h]>=Food2Divide){
          Culture[length(Culture)+1]<-0
          Reserve[length(Reserve)+1]<-IE
          Reserve[h]<-Reserve[h]-IE
        }
        
        
      }
      
      Death<-which(Reserve<0)
      if(length(Death)>0){
        Culture<-Culture[-Death]
        Reserve<-Reserve[-Death]
      }
      
      Death<-which(Culture>=DieEvent)
      if(length(Death)>0){
        Culture<-Culture[-Death]
        Reserve<-Reserve[-Death]
      }
      
      
      if(length(Culture)==0){
        Culture<-c(0)
      }
      
      Summary$Cells[i]<-length(Culture)
      Summary$Food[i]<-sum(Food)
      Summary$Time[i]<-i
      Summary$AvrAge[i]<-mean(Culture)
      Summary$EntropyAge[i]<-entropy(Culture,method = "ML")
      Summary$EntropyIE[i]<-entropy(Reserve,method = "ML")
      
      withProgress(message = 'Growing and killing units', value = 0, {
       incProgress(1, detail = paste("Iteration:", i))
       
      })
      
      
      if(Culture[1]!=0){
        Culture <- Culture + 1
      }
      Summary$FoodR<-Summary$Food*100/NutrientsSize 

    }    

    
    
    
        
    output$Plot <- renderPlot( { 
      
      p<-ggplot(Summary)+
        geom_line(aes(x=Time, y =Cells), colour="blue",size=2)+
        #geom_line(aes(x=Time, y= FoodRel), colour="green")+
        #geom_line(aes(x=Time, y=EntropyAge*EntropyIE*180),colour="violet")+
        xlim(1,iterations)+
        xlab("Time (AU)")+
        ylab("Number of Units")+
        theme_minimal(base_size = 20)+
        ggtitle("Number of Units")
        
      print(p)
      
    }) 
    
   
      
    output$Plot2 <- renderPlot( { 
      
      q<-ggplot(Summary)+
        
        geom_line(aes(x=Time, y=AvrAge), colour="red", size=2)+
        xlim(1,iterations)+
        xlab("Time (AU)")+
        ylab("Average Age (Time Units)")+
        ggtitle("Average Age of the Population")+
        theme_minimal(base_size = 20)
      
      print(q)
      
    }) 
    
    output$Plot3 <- renderPlot( { 
      
      q<-ggplot(Summary)+
        
        geom_line(aes(x=Time, y=FoodR), colour="green", size=2)+
        xlim(1,iterations)+
        xlab("Time (AU)")+
        ylab("% Food")+
        theme_minimal(base_size = 20)+
        ggtitle("Available Food")+
        ylim(0,100)
      
      print(q)
      
    })
    
    output$summary <- renderTable({
      Summary[,1:4]
    })
    


  })

  output$help <- renderUI({
    
    HTML(paste(
      "","","This numerical framework provides an exploratory tool to analyze how different factors affect the dynamic of a population of replicating entities.",
      "","<b>Total Food Supply:</b> Total amount of food available for the replicators to divide and maintain themselves","",
      "<b>Food Required to replicate:</b> Amount of internal energy (Food) that replicators need to adquire before they can divide",
      "","<b>Food intake/time unit:</b> Number of food units that replicators transform into internal energy every time unit. When the food is less abundant their ability to aquire food is reduced",
      "","<b>Programmed death at age:</b> Age at which the replicators commit suicide","","<b>Iterations:</b> Number of time events",
      "","<b>Internal Energy:</b> Starting internal energy","","<b>Senescence Factor:</b>How fast the replicators degrade with time (more degradation means more energy used for self-maintenance",
      
      sep = "</br>"
    ))
    
    
  })
  
  
  }

# Run the application 
shinyApp(ui = ui, server = server)

