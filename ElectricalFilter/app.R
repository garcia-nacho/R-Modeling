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
        
        tabsetPanel(
          tabPanel("Plot", plotOutput("Plot")),
          tabPanel("Table", tableOutput("table")),
          tabPanel("Summary", verbatimTextOutput("summary")),
          tabPanel("Info", htmlOutput("help"))
        )
        
        
        
        

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
    
    output$summary <- renderPrint({
      summary(Signal.Noise)
    })
    
    output$table <- renderTable({
      Signal.Noise 
    })
    


    
    output$help <- renderUI({
     HTML(paste("","","Electrical filters are combinations of electronic components that transform input signals into modified outputs and one of the most simple filters is the low-pass filter.", 
"","Canonical low-pass filters consist of a resistor and a capacitor connected in parallel (see the scheme on the left) and they have been traditionally used to remove high frequencies from sigmoidal inputs. In this very specific situation, the cut-off frequencies can be calculated from the resistance and capacitance values and the behavior of the filter is very well understood; however, the role of low-pass filters in noise suppression has not been fully explored.",
"",
"I became interested in the relationship between low-pass filters and noise-suppression because all biological pathways are very reliable systems which need to effectively distinguish between noise and signal to provide fidelity in the response. Unfortunately, I found that the differential equations that describe the behavior of low-pass filters are not able to integrate the noise in the system in a useful way so to circumvent that problem I developed this numerical simulation.", 
"","",
"I hope you like it.", sep = "</br>"))
    })
    

  })        
}
# Run the application 
shinyApp(ui = ui, server = server)

