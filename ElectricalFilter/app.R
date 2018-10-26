#  Numerical simulation for a low-pass filter
#  Nacho garcia 2018
#  garcia.nacho@gmail.com
#  GPL v3

#Library loading
library(shiny)
library(ggplot2)

# Define UI
ui <- fluidPage(
   
   # Application title
   titlePanel("Low-pass Filter Numerical Simulator"),
   
   # Sidebar 
   sidebarLayout(
      sidebarPanel(
        img(src='https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/RC_Divider.svg/200px-RC_Divider.svg.png', align = "center"),
        sliderInput("res", "R value (AU):", 1,100,50),
        sliderInput("cap", "C value (AU):", 1,100,50),
        sliderInput("noise", "Noise Level (AU):", 1,100,50),
        actionButton("do", "RUN", align = "center"),
        
        h6("By Nacho Garcia 2018", align = "left"),
        tags$a(href="https://github.com/garcia-nacho/R-Modeling/blob/master/ElectricalFilter/app.R", "Source Code")
        
        
        
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

  
    Noise <- as.data.frame(1:200)
    stoch<-input$noise * 0.002
    IN <- runif(200, 0, stoch)
    Signal.Noise <- cbind(Noise, IN)
    Signal.Noise$Out<-0.01
    r<- input$res *0.6
    c<-input$cap *0.3

    for (i in 2:200) {
      
      Signal.Noise$Out[i]<-((Signal.Noise$IN[i-1]-Signal.Noise$Out[i-1])/(r*c*(Signal.Noise$Out[i-1])))
      Signal.Noise$Out[i]<-Signal.Noise$Out[i-1]+Signal.Noise$Out[i]
    }
    
    colnames(Signal.Noise)<-c("Time(AU)","IN","Vout")

    output$Plot <- renderPlot( { 
      
      p<-ggplot(Signal.Noise)+
        geom_line(aes(x=`Time(AU)`, y=Vout), colour="red")+
        geom_line(aes(x=`Time(AU)`, y=IN), colour="black")+
        annotate("text", x=15, y=0.20, label= " Output", colour="red", size=5)+
        annotate("text", x=15, y=0.19, label= "Input", colour="black", size=5)+
        xlim(2,200)+
        ylim(0,0.2)+
        theme_minimal()
      
      print(p)
      
    })
  
   
  })        
    

}

# Run the application 
shinyApp(ui = ui, server = server)

