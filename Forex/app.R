#Implementation of a RNN for forex analysis
#Nacho Garcia 2018
#Based on the rnn library (https://github.com/bquast/rnn)


library(shiny)
library(ggplot2)
library(httr)
library(sigmoid)


# Define UI for application that draws a histogram
ui <- fluidPage(
   
   # Application title
   titlePanel("Recursive Neural Network Predicting Forex Values (November 2018)"),
   
   # 1. Open AUD/USD
   # 5. Open EUR/USD
   # 9. Open GBP/USD
   # 13.Open NZD/USD
   # 17.Open USD/CAD
   # 21.Open USD/CHF
   # 24.Open USD/JPY 
   
   # Sidebars
   sidebarLayout(
      sidebarPanel(
        
        selectInput("predX", "Pair to predict:",
                   choices =  c("AUD/USD" = 1,
                                   "EUR/USD" = 5,
                                   "GBP/USD" = 9,
                                   "NZD/USD"=13,
                                   "USD/CAD"=17,
                                   "USD/CHF"=21,
                                   "USD/JPY"=24), selected = 1),
        
        sliderInput("epoch", "Iterations:", 100,5000,1000),
        sliderInput("L", "Events Used for Training:", 500,3000,2000),
        sliderInput("nn", "Neurons in the RNN:", 4,128,12),
        sliderInput("lr", "Learning Rate:", 0.00001,0.01,0.0001),
        h6(""),
        actionButton("do", "Predict", align = "center"),
        h6("By Nacho Garcia 2018", align = "left"),
        tags$a(href="https://github.com/garcia-nacho/R-Modeling/blob/master/AgeModel/app.R", "Source Code")
        
      ),
      
      # Show a plot of the generated distribution
      mainPanel(tabsetPanel(id='conditioned',
                            tabPanel("Info", htmlOutput("help")),
                            tabPanel("Plots", plotOutput("Plot"), plotOutput("Plot2") ),
                            tabPanel("Values", tableOutput("summary"))
                            
                            
                            
      ))
      
      
      
      
      )
   )


# Define server logic required to draw a histogram
server <- function(input, output) {
  
  

  
  observeEvent(input$do, {
    
    Pred<-as.numeric(input$predX)
    epochD<-input$epoch
    L<-input$L
    lr<-input$lr
    nn<-input$nn
    
    GAP<-1
    
    
     #Pred<-5
    # epochD<-300
    # L<-3000
    # lr<-0.0001
    # nn<-12
    # 
    
    
    
    withProgress(message = 'Predicting', value = 0, {
      incProgress(1, detail = paste("Loading Pairs"))
      
  
      
      response <- GET(url="https://www.dropbox.com/s/v3ztdtxcbpdx9mn/dfBrute.RData?dl=1")
      load(rawConnection(response$content))
      rm(response)
      
      Values<-dfBrute[[Pred]]
      
      for (h in 1:length(dfBrute)) {
        dfBrute[[h]]$Open<-(dfBrute[[h]]$Open-min(dfBrute[[h]]$Open)) / (max(dfBrute[[h]]$Open)-min(dfBrute[[h]]$Open))
        dfBrute[[h]]$High<-(dfBrute[[h]]$High-min(dfBrute[[h]]$High)) / (max(dfBrute[[h]]$High)-min(dfBrute[[h]]$High))
        dfBrute[[h]]$Low<-(dfBrute[[h]]$Low-min(dfBrute[[h]]$Low)) / (max(dfBrute[[h]]$Low)-min(dfBrute[[h]]$Low))
        dfBrute[[h]]$Volume<-(dfBrute[[h]]$Volume-min(dfBrute[[h]]$Volume)) / (max(dfBrute[[h]]$Volume)-min(dfBrute[[h]]$Volume))
      }
      
      
    
    df<-array(data = NA, dim = c(1,nrow(dfBrute[[1]]),28))
    dummy<-0
    for (i in 1:length(dfBrute)) {
      
      df[1,,i+dummy]<-dfBrute[[i]]$Open
      df[1,,i+1+dummy]<-dfBrute[[i]]$High
      df[1,,i+2+dummy]<-dfBrute[[i]]$Low
      df[1,,i+3+dummy]<-dfBrute[[i]]$Volume
      dummy<-dummy+3
      
    }
    
    })
    
    
    #Training data
    Xt <- df[,1:L,]
    X.train <- array(Xt,dim=c(1,L,28))
    
    Yt <- df[,1:L+GAP,Pred]
    y.train <- matrix(Yt, ncol=L)
    
    ####RNN functions from rnn library
    
    init_r = function(model){
      if(model$network_type == "rnn"){
        init_rnn(model)
      } else if (model$network_type == "lstm"){
        init_lstm(model)
      }else if (model$network_type == "gru"){
        init_gru(model)
      }
    }
    
    #' @name init_rnn
    #' @title init_rnn
    #' @description Initialize the weight parameter for a rnn
    #' @param model the output model object
    #' @return the updated model
    
    init_rnn = function(model){
      
      # Storing layers states, filled with 0 for the moment
      model$store <- list()
      for(i in seq(length(model$synapse_dim) - 1)){
        model$store[[i]] <- array(0,dim = c(dim(model$last_layer_error)[1:2],model$synapse_dim[i+1]))
      }
      
      model$time_synapse            = list()
      model$recurrent_synapse       = list()
      model$bias_synapse            = list()
      
      #initialize neural network weights, stored in several lists
      for(i in seq(length(model$synapse_dim) - 1)){
        model$time_synapse[[i]] <- matrix(runif(n = model$synapse_dim[i]*model$synapse_dim[i+1], min=-1, max=1), nrow=model$synapse_dim[i])
      }
      for(i in seq(length(model$hidden_dim))){
        model$recurrent_synapse[[i]] <- matrix(runif(n = model$hidden_dim[i]*model$hidden_dim[i], min=-1, max=1), nrow=model$hidden_dim[i])
      }
      for(i in seq(length(model$synapse_dim) - 1)){
        model$bias_synapse[[i]] <- runif(model$synapse_dim[i+1],min=-0.1,max=0.1)
      }
      
      # add the update to the model list
      model$time_synapse_update = lapply(model$time_synapse,function(x){x*0})
      model$bias_synapse_update = lapply(model$bias_synapse,function(x){x*0})
      model$recurrent_synapse_update = lapply(model$recurrent_synapse,function(x){x*0})
      
      return(model)
    }
    
    #' @name init_lstm
    #' @title init_lstm
    #' @description Initialize the weight parameter for a lstm
    #' @param model the output model object
    #' @return the updated model
    
    init_lstm = function(model){
      # if(length(model$hidden_dim) != 1){stop("only one layer LSTM supported yet")}
      
      # Storing layers states, filled with 0 for the moment
      model$store <- list()
      model$time_synapse            = list()
      model$recurrent_synapse       = list()
      model$bias_synapse            = list()
      for(i in seq(length(model$hidden_dim))){
        # hidden output / cells / forget / input / gate / output
        model$store[[i]] = array(0,dim = c(dim(model$last_layer_error)[1:2],model$hidden_dim[1],6)) # 4D arrays !!! with dim()[4] = 6
        model$time_synapse[[i]] = array(runif(n = model$synapse_dim[i] * model$synapse_dim[i+1] * 4, min=-1, max=1),dim = c(model$synapse_dim[i], model$synapse_dim[i+1], 4))# 3D arrays with dim()[3] = 4
        model$recurrent_synapse[[i]] = array(runif(n = model$synapse_dim[i+1] * model$synapse_dim[i+1] * 4, min=-1, max=1),dim = c(model$synapse_dim[i+1], model$synapse_dim[i+1], 4))# 3D arrays with dim()[3] = 4
        model$bias_synapse[[i]] = array(runif(n = model$synapse_dim[i+1] * 4, min=-1, max=1),dim = c(model$synapse_dim[i+1], 4))#2D arrays with dim()[2] = 4
      }
      
      model$store[[length(model$store) + 1]] = array(0,dim = c(dim(model$last_layer_error)[1:2],model$output_dim)) # final output layer
      model$time_synapse[[length(model$time_synapse) + 1]] = array(runif(n = model$hidden_dim[length(model$hidden_dim)] * model$output_dim, min=-1, max=1),dim = c(model$hidden_dim[length(model$hidden_dim)], model$output_dim)) # 4D arrays !!!
      model$bias_synapse[[length(model$bias_synapse) + 1]] = runif(model$output_dim,min=-0.1,max=0.1)
      
      # add the update to the model list
      model$time_synapse_update = lapply(model$time_synapse,function(x){x*0})
      model$bias_synapse_update = lapply(model$bias_synapse,function(x){x*0})
      model$recurrent_synapse_update = lapply(model$recurrent_synapse,function(x){x*0})
      
      return(model)
    }
    
    #' @name init_gru
    #' @title init_gru
    #' @description Initialize the weight parameter for a gru
    #' @param model the output model object
    #' @return the updated model
    
    init_gru = function(model){
      
      # Storing layers states, filled with 0 for the moment
      model$store <- list()
      model$time_synapse            = list()
      model$recurrent_synapse       = list()
      model$bias_synapse            = list()
      for(i in seq(length(model$hidden_dim))){
        # hidden output / z / r / h
        model$store[[i]] = array(0,dim = c(dim(model$last_layer_error)[1:2],model$hidden_dim[1],4)) # 4D arrays !!! with dim()[4] = 4
        model$time_synapse[[i]] = array(runif(n = model$synapse_dim[i] * model$synapse_dim[i+1] * 3, min=-1, max=1),dim = c(model$synapse_dim[i], model$synapse_dim[i+1], 3))# 3D arrays with dim()[3] = 3
        model$recurrent_synapse[[i]] = array(runif(n = model$synapse_dim[i+1] * model$synapse_dim[i+1] * 3, min=-1, max=1),dim = c(model$synapse_dim[i+1], model$synapse_dim[i+1], 3))# 3D arrays with dim()[3] = 3
        model$bias_synapse[[i]] = array(runif(n = model$synapse_dim[i+1] * 3, min=-1, max=1),dim = c(model$synapse_dim[i+1], 3))#2D arrays with dim()[2] = 3
      }
      
      model$store[[length(model$store) + 1]] = array(0,dim = c(dim(model$last_layer_error)[1:2],model$output_dim)) # final output layer
      model$time_synapse[[length(model$time_synapse) + 1]] = array(runif(n = model$hidden_dim[length(model$hidden_dim)] * model$output_dim, min=-1, max=1),dim = c(model$hidden_dim[length(model$hidden_dim)], model$output_dim)) # 4D arrays !!!
      model$bias_synapse[[length(model$bias_synapse) + 1]] = runif(model$output_dim,min=-0.1,max=0.1)
      
      # add the update to the model list
      model$time_synapse_update = lapply(model$time_synapse,function(x){x*0})
      model$bias_synapse_update = lapply(model$bias_synapse,function(x){x*0})
      model$recurrent_synapse_update = lapply(model$recurrent_synapse,function(x){x*0})
      
      return(model)
    }
    
    #' @name backprop_r
    #' @title backprop_r
    #' @description backpropagate the error in a model object
    #' @param model the output model object
    #' @param a the input of this learning batch
    #' @param c the output of this learning batch
    #' @param j the indexes of the sample in the current batch
    #' @param ... argument to be passed to method
    #' @return the updated model
    
    backprop_r = function(model,a,c,j,...){
      if(model$network_type == "rnn"){
        backprop_rnn(model,a,c,j,...)
      } else if (model$network_type == "lstm"){
        backprop_lstm(model,a,c,j,...)
      } else if (model$network_type == "gru"){
        backprop_gru(model,a,c,j,...)
      }else{
        stop("network_type_unknown for the backprop")
      }
    }
    
    #' @name backprop_rnn
    #' @title backprop_rnn
    #' @description backpropagate the error in a model object of type rnn
    #' @param model the output model object
    #' @param a the input of this learning batch
    #' @param c the output of this learning batch
    #' @param j the indexes of the sample in the current batch
    #' @param ... argument to be passed to method
    #' @return the updated model
    
    backprop_rnn = function(model,a,c,j,...){
      
      # store errors
      model$last_layer_error[j,,] = c - model$store[[length(model$store)]][j,,,drop=F]
      model$last_layer_delta[j,,] = model$last_layer_error[j,,,drop = F] * sigmoid_output_to_derivative(model$store[[length(model$store)]][j,,,drop=F])
      
      if(model$seq_to_seq_unsync){
        model$last_layer_error[j,1:(model$time_dim_input - 1),] = 0
        model$last_layer_delta[j,1:(model$time_dim_input - 1),] = 0
      }
      
      
      model$error[j,model$current_epoch] <- apply(model$last_layer_error[j,,,drop=F],1,function(x){sum(abs(x))})
      
      # init futur layer delta, here because there is no layer delta at time_dim+1
      future_layer_delta  = list()
      for(i in seq(length(model$hidden_dim))){
        future_layer_delta[[i]] <- matrix(0,nrow=length(j), ncol = model$hidden_dim[i])
      }
      
      # Weight iteration,
      for (position in model$time_dim:1) {
        
        # input states
        x            = array(a[,position,],dim=c(length(j),model$input_dim))
        # error at output layer
        layer_up_delta = array(model$last_layer_delta[j,position,],dim=c(length(j),model$output_dim))
        
        for(i in (length(model$store)):1){
          if(i != 1){ # need update for time and recurrent synapse
            layer_current      = array(model$store[[i-1]][j,position,],dim=c(length(j),model$hidden_dim[i-1]))
            if(position != 1){
              prev_layer_current = array(model$store[[i-1]][j,position - 1,],dim=c(length(j),model$hidden_dim[i-1]))
            }else{
              prev_layer_current = array(0,dim=c(length(j),model$hidden_dim[i-1]))
            }
            # error at hidden layers
            layer_current_delta = (future_layer_delta[[i-1]] %*% t(model$recurrent_synapse[[i-1]]) + layer_up_delta %*% t(model$time_synapse[[i]])) *
              sigmoid_output_to_derivative(layer_current)
            model$time_synapse_update[[i]] = model$time_synapse_update[[i]] + t(layer_current) %*% layer_up_delta
            model$bias_synapse_update[[i]] = model$bias_synapse_update[[i]] + colMeans(layer_up_delta)
            model$recurrent_synapse_update[[i-1]] = model$recurrent_synapse_update[[i-1]] + t(prev_layer_current) %*% layer_current_delta
            layer_up_delta = layer_current_delta
            future_layer_delta[[i-1]] = layer_current_delta
          }else{ # need only update for time synapse
            model$time_synapse_update[[i]] = model$time_synapse_update[[i]] + t(x) %*% layer_up_delta
          }
        }
      } # end position back prop loop
      return(model)
    }
    
    #' @name backprop_lstm
    #' @title backprop_lstm
    #' @description backpropagate the error in a model object of type rlstm
    #' @importFrom sigmoid tanh_output_to_derivative
    #' @param model the output model object
    #' @param a the input of this learning batch
    #' @param c the output of this learning batch
    #' @param j the indexes of the sample in the current batch
    #' @param ... argument to be passed to method
    #' @return the updated model
    
    backprop_lstm = function(model,a,c,j,...){
      
      # store errors
      model$last_layer_error[j,,] = c - model$store[[length(model$store)]][j,,,drop=F]
      model$last_layer_delta[j,,] = model$last_layer_error[j,,,drop = F] * sigmoid_output_to_derivative(model$store[[length(model$store)]][j,,,drop=F])
      
      if(model$seq_to_seq_unsync){
        model$last_layer_error[j,1:(model$time_dim_input - 1),] = 0
        model$last_layer_delta[j,1:(model$time_dim_input - 1),] = 0
      }
      model$error[j,model$current_epoch] <- apply(model$last_layer_error[j,,,drop=F],1,function(x){sum(abs(x))})
      
      future_layer_cell_delta = list()
      future_layer_hidden_delta = list()
      for(i in seq(length(model$hidden_dim))){
        future_layer_cell_delta[[i]] = matrix(0, nrow = length(j), ncol = model$hidden_dim[i]) # 4 to actualize
        future_layer_hidden_delta[[i]] = matrix(0, nrow = length(j), ncol = model$hidden_dim[i]) # 2, to actualize
      }
      
      
      for (position in model$time_dim:1) {
        # error at output layer
        layer_up_delta = array(model$last_layer_delta[j,position,],dim=c(length(j),model$output_dim))
        
        # first the last layer to update the layer_up_delta
        i = length(model$hidden_dim)
        layer_hidden = array(model$store[[i]][j,position,,1],dim=c(length(j),model$hidden_dim[i]))
        # output layer update
        model$time_synapse_update[[i+1]]   = model$time_synapse_update[[i+1]]   + (t(layer_hidden) %*% layer_up_delta)
        model$bias_synapse_update[[i+1]]   = model$bias_synapse_update[[i+1]]   + colMeans(layer_up_delta)
        # lstm hidden delta
        layer_up_delta = (layer_up_delta %*% t(model$time_synapse_update[[i+1]])) * sigmoid_output_to_derivative(layer_hidden) + future_layer_hidden_delta[[i]] # 1 and 3
        
        for(i in length(model$hidden_dim):1){
          # x: input of the layer
          if(i == 1){
            x = array(a[,position,],dim=c(length(j),model$input_dim))
          }else{
            x = array(model$store[[i - 1]][j,position,,1],dim=c(length(j),model$synapse_dim[i]))
          }
          layer_hidden = array(model$store[[i]][j,position,,1],dim=c(length(j),model$hidden_dim[i]))
          layer_cell = array(model$store[[i]][j,position,,2],dim=c(length(j), model$hidden_dim[i]))
          if(position != 1){
            prev_layer_hidden =array(model$store[[i]][j,position-1,,1],dim=c(length(j),model$hidden_dim[i]))
            preview_layer_cell = array(model$store[[i]][j,position-1,,2],dim=c(length(j), model$hidden_dim[i]))
          }else{
            prev_layer_hidden =array(0,dim=c(length(j),model$hidden_dim[i]))
            preview_layer_cell = array(0,dim=c(length(j), model$hidden_dim[i]))
          }
          
          layer_f = array(model$store[[i]][j,position,,3],dim=c(length(j), model$hidden_dim[i]))
          layer_i = array(model$store[[i]][j,position,,4],dim=c(length(j), model$hidden_dim[i]))
          layer_c = array(model$store[[i]][j,position,,5],dim=c(length(j), model$hidden_dim[i]))
          layer_o = array(model$store[[i]][j,position,,6],dim=c(length(j), model$hidden_dim[i]))
          
          # lstm cell delta
          # layer_cell_delta = (layer_hidden_delta * layer_o) + future_layer_cell_delta    # 5 then 8 (skip 7 as no tanh)
          # layer_o_delta_post_activation = layer_hidden_delta *  layer_cell # 6 (skip 7 as no tanh)
          layer_cell_delta = (layer_up_delta * layer_o)* tanh_output_to_derivative(layer_cell) + future_layer_cell_delta[[i]]    # 5, 7 then 8
          layer_o_delta_post_activation = layer_up_delta *  tanh(layer_cell) # 6 
          
          layer_c_delta_post_activation = layer_cell_delta * layer_i    # 9
          layer_i_delta_post_activation = layer_cell_delta * layer_c    # 10
          
          layer_f_delta_post_activation = layer_cell_delta * preview_layer_cell # 12
          future_layer_cell_delta[[i]] = layer_cell_delta * layer_f # 11
          
          layer_o_delta_pre_activation = layer_o_delta_post_activation * sigmoid_output_to_derivative(layer_o) # 13
          layer_c_delta_pre_activation = layer_c_delta_post_activation * tanh_output_to_derivative(layer_c) # 14
          layer_i_delta_pre_activation = layer_i_delta_post_activation * sigmoid_output_to_derivative(layer_i) # 15
          layer_f_delta_pre_activation = layer_f_delta_post_activation * sigmoid_output_to_derivative(layer_f) # 16
          # 
          
          
          # let's update all our weights so we can try again
          model$recurrent_synapse_update[[i]][,,1] = model$recurrent_synapse_update[[i]][,,1] + t(prev_layer_hidden) %*% layer_f_delta_post_activation
          model$recurrent_synapse_update[[i]][,,2] = model$recurrent_synapse_update[[i]][,,2] + t(prev_layer_hidden) %*% layer_i_delta_post_activation
          model$recurrent_synapse_update[[i]][,,3] = model$recurrent_synapse_update[[i]][,,3] + t(prev_layer_hidden) %*% layer_c_delta_post_activation
          model$recurrent_synapse_update[[i]][,,4] = model$recurrent_synapse_update[[i]][,,4] + t(prev_layer_hidden) %*% layer_o_delta_post_activation
          model$time_synapse_update[[i]][,,1] = model$time_synapse_update[[i]][,,1] + t(x) %*% layer_f_delta_post_activation
          model$time_synapse_update[[i]][,,2] = model$time_synapse_update[[i]][,,2] + t(x) %*% layer_i_delta_post_activation
          model$time_synapse_update[[i]][,,3] = model$time_synapse_update[[i]][,,3] + t(x) %*% layer_c_delta_post_activation
          model$time_synapse_update[[i]][,,4] = model$time_synapse_update[[i]][,,4] + t(x) %*% layer_o_delta_post_activation
          model$bias_synapse_update[[i]][,1] = model$bias_synapse_update[[i]][,1] + colMeans(layer_f_delta_post_activation)
          model$bias_synapse_update[[i]][,2] = model$bias_synapse_update[[i]][,2] + colMeans(layer_i_delta_post_activation)
          model$bias_synapse_update[[i]][,3] = model$bias_synapse_update[[i]][,3] + colMeans(layer_c_delta_post_activation)
          model$bias_synapse_update[[i]][,4] = model$bias_synapse_update[[i]][,4] + colMeans(layer_o_delta_post_activation)
          
          layer_f_delta_pre_weight = layer_f_delta_pre_activation %*% t(array(model$recurrent_synapse[[i]][,,1],dim=c(dim(model$recurrent_synapse[[i]])[1:2]))) # 20
          layer_i_delta_pre_weight = layer_i_delta_pre_activation %*% t(array(model$recurrent_synapse[[i]][,,2],dim=c(dim(model$recurrent_synapse[[i]])[1:2]))) # 19
          layer_c_delta_pre_weight = layer_c_delta_pre_activation %*% t(array(model$recurrent_synapse[[i]][,,3],dim=c(dim(model$recurrent_synapse[[i]])[1:2]))) # 18
          layer_o_delta_pre_weight = layer_o_delta_pre_activation %*% t(array(model$recurrent_synapse[[i]][,,4],dim=c(dim(model$recurrent_synapse[[i]])[1:2]))) # 17
          future_layer_hidden_delta[[i]] = layer_o_delta_pre_weight + layer_c_delta_pre_weight + layer_i_delta_pre_weight + layer_f_delta_pre_weight # 21
          
          layer_f_delta_pre_weight = layer_f_delta_pre_activation %*% t(array(model$time_synapse[[i]][,,1],dim=c(dim(model$time_synapse[[i]])[1:2]))) # 20
          layer_i_delta_pre_weight = layer_i_delta_pre_activation %*% t(array(model$time_synapse[[i]][,,2],dim=c(dim(model$time_synapse[[i]])[1:2]))) # 19
          layer_c_delta_pre_weight = layer_c_delta_pre_activation %*% t(array(model$time_synapse[[i]][,,3],dim=c(dim(model$time_synapse[[i]])[1:2]))) # 18
          layer_o_delta_pre_weight = layer_o_delta_pre_activation %*% t(array(model$time_synapse[[i]][,,4],dim=c(dim(model$time_synapse[[i]])[1:2]))) # 17
          layer_up_delta = layer_o_delta_pre_weight + layer_c_delta_pre_weight + layer_i_delta_pre_weight + layer_f_delta_pre_weight # 21
        }
      }
      
      # future_layers_delta = list()
      # for(i in seq(length(model$hidden_dim))){
      #   future_layers_delta[[i]] = array(0,dim=c(length(j),model$hidden_dim[i],4))
      # }
      # 
      # for (position in model$time_dim:1) {
      #   
      #   # input states
      #   x            = array(a[,position,],dim=c(length(j),model$input_dim))
      #   # error at output layer
      #   layer_up_delta = array(model$last_layer_delta[j,position,],dim=c(length(j),model$output_dim))
      #   
      #   for(i in (length(model$store)):1){
      #     if(i != 1){ # need update for time and recurrent synapse
      #       layer_current      = array(model$store[[i-1]][j,position,],dim=c(length(j),model$hidden_dim[i-1]))
      #       if(position != 1){
      #         prev_layer_current = array(model$store[[i-1]][j,position - 1,],dim=c(length(j),model$hidden_dim[i-1]))
      #       }else{
      #         prev_layer_current = array(0,dim=c(length(j),model$hidden_dim[i-1]))
      #       }
      #     }
      #     if(i == length(model$store)){
      #       # error at hidden layer
      #       future_layers_delta[[i-1]][,,1] = (future_layers_delta[[i-1]][,,1] %*% t(model$recurrent_synapse[[i-1]][,,1]) + layer_up_delta %*% t(model$time_synapse_ouput)) *
      #         sigmoid_output_to_derivative(layer_current)
      #       future_layers_delta[[i-1]][,,2] = (future_layers_delta[[i-1]][,,2] %*% t(model$recurrent_synapse[[i-1]][,,2]) + layer_up_delta %*% t(model$time_synapse_ouput)) *
      #         sigmoid_output_to_derivative(layer_current)
      #       future_layers_delta[[i-1]][,,3] = (future_layers_delta[[i-1]][,,3] %*% t(model$recurrent_synapse[[i-1]][,,3]) + layer_up_delta %*% t(model$time_synapse_ouput)) *
      #         sigmoid_output_to_derivative(layer_current)
      #       future_layers_delta[[i-1]][,,4] = (future_layers_delta[[i-1]][,,4] %*% t(model$recurrent_synapse[[i-1]][,,4]) + layer_up_delta %*% t(model$time_synapse_ouput)) *
      #         sigmoid_output_to_derivative(layer_current)
      #       
      #       # let's update all our weights so we can try again
      #       model$time_synapse_ouput_update   = model$time_synapse_ouput_update   + t(layer_current)      %*% layer_up_delta
      #       model$recurrent_synapse_update[[i-1]][,,1]  = model$recurrent_synapse_update[[i-1]][,,1] + t(prev_layer_current) %*% future_layers_delta[[i-1]][,,1]
      #       model$recurrent_synapse_update[[i-1]][,,2]  = model$recurrent_synapse_update[[i-1]][,,2] + t(prev_layer_current) %*% future_layers_delta[[i-1]][,,2]
      #       model$recurrent_synapse_update[[i-1]][,,3]  = model$recurrent_synapse_update[[i-1]][,,3] + t(prev_layer_current) %*% future_layers_delta[[i-1]][,,3]
      #       model$recurrent_synapse_update[[i-1]][,,4]  = model$recurrent_synapse_update[[i-1]][,,4] + t(prev_layer_current) %*% future_layers_delta[[i-1]][,,4]
      #       
      #       model$bias_synapse_ouput_update = model$bias_synapse_ouput_update + colMeans(layer_up_delta)
      # 
      #       model$bias_synapse_update[[i-1]] = model$bias_synapse_update[[i-1]] + apply(future_layers_delta[[i-1]],2:3,mean)
      #       
      #     } else if(i == 1){
      #       model$time_synapse_update[[i]][,,1] = model$time_synapse_update[[i]][,,1] + t(x) %*% future_layers_delta[[i]][,,1]
      #       model$time_synapse_update[[i]][,,2] = model$time_synapse_update[[i]][,,2] + t(x) %*% future_layers_delta[[i]][,,2]
      #       model$time_synapse_update[[i]][,,3] = model$time_synapse_update[[i]][,,3] + t(x) %*% future_layers_delta[[i]][,,3]
      #       model$time_synapse_update[[i]][,,4] = model$time_synapse_update[[i]][,,4] + t(x) %*% future_layers_delta[[i]][,,4]
      #     }
      #   }
      # }
      return(model)
    }
    
    #' @name backprop_gru
    #' @title backprop_gru
    #' @description backpropagate the error in a model object of type gru
    #' @importFrom sigmoid tanh_output_to_derivative
    #' @param model the output model object
    #' @param a the input of this learning batch
    #' @param c the output of this learning batch
    #' @param j the indexes of the sample in the current batch
    #' @param ... argument to be passed to method
    #' @return the updated model
    
    backprop_gru = function(model,a,c,j,...){
      
      # store errors
      model$last_layer_error[j,,] = c - model$store[[length(model$store)]][j,,,drop=F]
      model$last_layer_delta[j,,] = model$last_layer_error[j,,,drop = F] * sigmoid_output_to_derivative(model$store[[length(model$store)]][j,,,drop=F])
      
      # many_to_one
      if(model$seq_to_seq_unsync){
        model$last_layer_error[j,1:(model$time_dim_input - 1),] = 0
        model$last_layer_delta[j,1:(model$time_dim_input - 1),] = 0
      }
      model$error[j,model$current_epoch] <- apply(model$last_layer_error[j,,,drop=F],1,function(x){sum(abs(x))})
      
      future_layer_hidden_delta = list()
      for(i in seq(length(model$hidden_dim))){
        future_layer_hidden_delta[[i]] = matrix(0, nrow = length(j), ncol = model$hidden_dim[i]) # 2, to actualize
      }
      
      
      for (position in model$time_dim:1) {
        # error at output layer
        layer_up_delta = array(model$last_layer_delta[j,position,],dim=c(length(j),model$output_dim))
        
        # first the last layer to update the layer_up_delta
        i = length(model$hidden_dim)
        layer_hidden = array(model$store[[i]][j,position,,1],dim=c(length(j),model$hidden_dim[i]))
        # output layer update
        model$time_synapse_update[[i+1]]   = model$time_synapse_update[[i+1]]   + (t(layer_hidden) %*% layer_up_delta)
        model$bias_synapse_update[[i+1]]   = model$bias_synapse_update[[i+1]]   + colMeans(layer_up_delta)
        # lstm hidden delta
        layer_up_delta = (layer_up_delta %*% t(model$time_synapse_update[[i+1]])) * sigmoid_output_to_derivative(layer_hidden) + future_layer_hidden_delta[[i]] # 1 and 3
        
        for(i in length(model$hidden_dim):1){
          # x: input of the layer
          if(i == 1){
            x = array(a[,position,],dim=c(length(j),model$input_dim))
          }else{
            x = array(model$store[[i - 1]][j,position,,1],dim=c(length(j),model$synapse_dim[i]))
          }
          layer_hidden = array(model$store[[i]][j,position,,1],dim=c(length(j),model$hidden_dim[i]))
          if(position != 1){
            prev_layer_hidden =array(model$store[[i]][j,position-1,,1],dim=c(length(j),model$hidden_dim[i]))
          }else{
            prev_layer_hidden =array(0,dim=c(length(j),model$hidden_dim[i]))
          }
          
          layer_z = array(model$store[[i]][j,position,,2],dim=c(length(j), model$hidden_dim[i]))
          layer_r = array(model$store[[i]][j,position,,3],dim=c(length(j), model$hidden_dim[i]))
          layer_h = array(model$store[[i]][j,position,,4],dim=c(length(j), model$hidden_dim[i]))
          
          layer_hidden_delta = layer_up_delta + future_layer_hidden_delta[[i]] #3
          layer_h_delta_post_activation = layer_hidden_delta *  layer_z # 6 
          layer_h_delta_pre_activation = layer_h_delta_post_activation * tanh_output_to_derivative(layer_h) # 6 bis
          layer_z_delta_post_split = layer_hidden_delta *  layer_h # 7 
          
          layer_z_delta_post_1_minus = layer_hidden_delta *  prev_layer_hidden # 9 
          layer_hidden_delta = layer_hidden_delta * (1 - layer_z) # 8
          
          layer_z_delta_post_activation = (1 - layer_z_delta_post_1_minus) # 10
          layer_z_delta_pre_activation = layer_z_delta_post_activation*  sigmoid_output_to_derivative(layer_z) # 10 bis
          layer_z_delta_pre_weight_h = (layer_z_delta_pre_activation %*% t(model$recurrent_synapse[[i]][,,1]) ) # 14 
          layer_z_delta_pre_weight_x = (layer_z_delta_pre_activation %*% array(t(model$time_synapse[[i]][,,1]),dim = dim(model$time_synapse[[i]])[2:1])) # 14 
          # let's update all our weights so we can try again
          model$recurrent_synapse_update[[i]][,,1] = model$recurrent_synapse_update[[i]][,,1] + t(prev_layer_hidden) %*% layer_z_delta_post_activation
          model$time_synapse_update[[i]][,,1] = model$time_synapse_update[[i]][,,1] + t(x) %*% layer_z_delta_post_activation
          model$bias_synapse_update[[i]][,1] = model$bias_synapse_update[[i]][,1] + colMeans(layer_z_delta_post_activation)
          
          layer_h_delta_pre_weight_h = (layer_h_delta_pre_activation %*% t(model$recurrent_synapse[[i]][,,3]))# 13 
          layer_h_delta_pre_weight_x = ( layer_h_delta_pre_activation %*% array(t(model$time_synapse[[i]][,,3]),dim = dim(model$time_synapse[[i]])[2:1])) # 13 
          # let's update all our weights so we can try again
          model$recurrent_synapse_update[[i]][,,3] = model$recurrent_synapse_update[[i]][,,3] + t(prev_layer_hidden * layer_r) %*% layer_h_delta_post_activation
          model$time_synapse_update[[i]][,,3] = model$time_synapse_update[[i]][,,3] + t(x) %*% layer_h_delta_post_activation
          model$bias_synapse_update[[i]][,3] = model$bias_synapse_update[[i]][,3] + colMeans(layer_h_delta_post_activation)
          
          layer_r_delta_post_activation = prev_layer_hidden * layer_h_delta_pre_weight_h # 15
          layer_r_delta_pre_activation = layer_r_delta_post_activation * sigmoid_output_to_derivative(layer_r) # 15 bis
          layer_hidden_delta = layer_hidden_delta + layer_r * layer_h_delta_pre_weight_h # 12
          
          layer_r_delta_pre_weight_h = (layer_r_delta_pre_activation %*% t(model$recurrent_synapse[[i]][,,2])) # 17 
          layer_r_delta_pre_weight_x = (layer_r_delta_post_activation %*% array(t(model$time_synapse[[i]][,,2]),dim = dim(model$time_synapse[[i]])[2:1])) # 17 
          # let's update all our weights so we can try again
          model$recurrent_synapse_update[[i]][,,2] = model$recurrent_synapse_update[[i]][,,2] + t(prev_layer_hidden) %*% layer_r_delta_post_activation
          model$time_synapse_update[[i]][,,2] = model$time_synapse_update[[i]][,,2] + t(x) %*% layer_r_delta_post_activation
          model$bias_synapse_update[[i]][,2] = model$bias_synapse_update[[i]][,2] + colMeans(layer_r_delta_post_activation)
          
          layer_r_and_z_delta_pre_weight_h = layer_r_delta_pre_weight_h + layer_z_delta_pre_weight_h # 19
          layer_r_and_z_delta_pre_weight_x = layer_r_delta_pre_weight_x + layer_z_delta_pre_weight_x # 19
          
          future_layer_hidden_delta[[i]] = layer_hidden_delta + layer_r_and_z_delta_pre_weight_h # 23
          
          layer_up_delta = layer_r_and_z_delta_pre_weight_x + layer_h_delta_pre_weight_x # 22
        }
      }
      return(model)
    }
    
    
    update_r = function(model){
      if(model$update_rule == "sgd"){
        update_sgd(model)
      }else if(model$update_rule == "adagrad"){
        update_adagrad(model)
      }else{
        stop("update_rule unknown")
      }
    }
    
    
    #' @name update_sgd
    #' @title update_sgd
    #' @description Apply the update with stochastic gradient descent
    #' @param model the output model object
    #' @return the updated model
    
    update_sgd = function(model){
      
      if(!is.null(model$clipping)){ # should we clippe the update or the weight, the update will make more sens as the weight lead to killed units
        clipping = function(x){
          x[is.nan(x)] = runif(sum(is.nan(x)),-1,1)
          x[x > model$clipping] = model$clipping
          x[x < -model$clipping] = - model$clipping
          return(x)
        }
        model$recurrent_synapse_update  = lapply(model$recurrent_synapse_update,clipping)
        model$time_synapse_update       = lapply(model$time_synapse_update,clipping)
        model$bias_synapse_update       = lapply(model$bias_synapse_update, clipping)
      }
      
      for(i in seq(length(model$time_synapse))){
        model$time_synapse[[i]] <- model$time_synapse[[i]] + model$time_synapse_update[[i]]
        model$bias_synapse[[i]] <- model$bias_synapse[[i]] + model$bias_synapse_update[[i]]
      }
      for(i in seq(length(model$recurrent_synapse))){
        model$recurrent_synapse[[i]] <- model$recurrent_synapse[[i]] + model$recurrent_synapse_update[[i]]
      }
      
      # Initializing the update with the momentum
      model$time_synapse_update = lapply(model$time_synapse_update,function(x){x* model$momentum})
      model$bias_synapse_update = lapply(model$bias_synapse_update,function(x){x* model$momentum})
      model$recurrent_synapse_update = lapply(model$recurrent_synapse_update,function(x){x* model$momentum})
      
      return(model)
    }
    
    
    #' @name update_adagrad
    #' @title update_adagrad
    #' @description Apply the update with adagrad, not working yet
    #' @param model the output model object
    #' @return the updated model
    
    update_adagrad = function(model){
      ## not working yet, inspiration here:
      ## https://www.quora.com/What-are-differences-between-update-rules-like-AdaDelta-RMSProp-AdaGrad-and-AdaM
      if(!is.null(model$clipping)){ # should we clippe the update or the weight, the update will make more sens as the weight lead to killed units
        clipping = function(x){
          x[is.nan(x)] = runif(sum(is.nan(x)),-1,1)
          x[x > model$clipping] = model$clipping
          x[x < -model$clipping] = - model$clipping
          return(x)
        }
        model$recurrent_synapse_update  = lapply(model$recurrent_synapse_update,clipping)
        model$time_synapse_update       = lapply(model$time_synapse_update,clipping)
        model$bias_synapse_update       = lapply(model$bias_synapse_update, clipping)
      }
      
      if(is.null(model$recurrent_synapse_update_old)){ # not really the old update but just a store of the old
        model$recurrent_synapse_update_old = lapply(model$recurrent_synapse_update,function(x){x*0})
        model$time_synapse_update_old = lapply(model$time_synapse_update,function(x){x*0})
        # model$bias_synapse_update_old = lapply(model$bias_synapse_update,function(x){x*0}) # the bias stay the same, we only apply it on the weight
      }
      
      for(i in seq(length(model$time_synapse))){
        model$time_synapse_update_old[[i]] <- model$time_synapse_update_old[[i]] + model$time_synapse_update[[i]]
        # model$bias_synapse_old[[i]] <- model$bias_synapse[[i]] + model$bias_synapse_update[[i]]
      }
      for(i in seq(length(model$recurrent_synapse))){
        model$recurrent_synapse_update_old[[i]] <- model$recurrent_synapse_update_old[[i]] + model$recurrent_synapse_update[[i]]
      }
      
      for(i in seq(length(model$time_synapse))){
        model$time_synapse[[i]] <- model$time_synapse[[i]] + model$learningrate * model$time_synapse_update[[i]] / (model$time_synapse_update_old[[i]] + 0.000000001)
        model$bias_synapse[[i]] <- model$bias_synapse[[i]] + model$bias_synapse_update[[i]]
      }
      for(i in seq(length(model$recurrent_synapse))){
        model$recurrent_synapse[[i]] <- model$recurrent_synapse[[i]] + model$learningrate * model$recurrent_synapse_update[[i]] / (model$recurrent_synapse_update_old[[i]] + 0.000000001)
      }
      
      return(model)
    }
    
    #' @name epoch_print
    #' @export
    #' @title epoch printing for trainr
    #' @description Print the error adn learning rate at each epoch of the trainr learning, called in epoch_function
    #' @param model the output model object
    #' @return nothing
    
    epoch_print = function(model){
      message(paste0("Trained epoch: ",model$current_epoch," - Learning rate: ",model$learningrate))
      message(paste0("Epoch error: ",colMeans(model$error)[model$current_epoch]))
      return(model)
    }
    
    #' @name epoch_annealing
    #' @export
    #' @title epoch annealing
    #' @description Apply the learning rate decay to the learning rate, called in epoch_model_function
    #' @param model the output model object
    #' @return the updated model
    
    epoch_annealing = function(model){
      model$learningrate = model$learningrate * model$learningrate_decay
      return(model)
    }
    
    #' @name loss_L1
    #' @export
    #' @title L1 loss
    #' @description Apply the learning rate to the weight update, vocabulary to verify !!
    #' @param model the output model object
    #' @return the updated model
    
    loss_L1 = function(model){
      if(model$network_type == "rnn"){
        model$time_synapse_update = lapply(model$time_synapse_update,function(x){x* model$learningrate})
        model$bias_synapse_update = lapply(model$bias_synapse_update,function(x){x* model$learningrate})
        model$recurrent_synapse_update = lapply(model$recurrent_synapse_update,function(x){x* model$learningrate})
      } else if(model$network_type == "lstm"){
        model$recurrent_synapse_update  = lapply(model$recurrent_synapse_update,function(x){x * model$learningrate})
        model$time_synapse_update       = lapply(model$time_synapse_update,function(x){x * model$learningrate})
        model$bias_synapse_update       = lapply(model$bias_synapse_update, function(x){x * model$learningrate})
      } else if(model$network_type == "gru"){
        model$recurrent_synapse_update  = lapply(model$recurrent_synapse_update,function(x){x * model$learningrate})
        model$time_synapse_update       = lapply(model$time_synapse_update,function(x){x * model$learningrate})
        model$bias_synapse_update       = lapply(model$bias_synapse_update, function(x){x * model$learningrate})
      }
      return(model)
    }
    
    
    #' @name predictr
    #' @export
    #' @importFrom stats runif
    #' @importFrom sigmoid sigmoid
    #' @title Recurrent Neural Network
    #' @description predict the output of a RNN model
    #' @param model output of the trainr function
    #' @param X array of input values, dim 1: samples, dim 2: time, dim 3: variables (could be 1 or more, if a matrix, will be coerce to array)
    #' @param hidden should the function output the hidden units states
    #' @param real_output option used when the function in called inside trainr, do not drop factor for 2 dimension array output and other actions. Let it to TRUE, the default, to let the function take care of the data.
    #' @param ... arguments to pass on to sigmoid function
    #' @return array or matrix of predicted values
    #' @examples
    #' \dontrun{ 
    #' # create training numbers
    #' X1 = sample(0:127, 10000, replace=TRUE)
    #' X2 = sample(0:127, 10000, replace=TRUE)
    #' 
    #' # create training response numbers
    #' Y <- X1 + X2
    #' 
    #' # convert to binary
    #' X1 <- int2bin(X1)
    #' X2 <- int2bin(X2)
    #' Y  <- int2bin(Y)
    #' 
    #' # Create 3d array: dim 1: samples; dim 2: time; dim 3: variables.
    #' X <- array( c(X1,X2), dim=c(dim(X1),2) )
    #' 
    #' # train the model
    #' model <- trainr(Y=Y[,dim(Y)[2]:1],
    #'                 X=X[,dim(X)[2]:1,],
    #'                 learningrate   =  1,
    #'                 hidden_dim     = 16 )
    #'              
    #' # create test inputs
    #' A1 = int2bin( sample(0:127, 7000, replace=TRUE) )
    #' A2 = int2bin( sample(0:127, 7000, replace=TRUE) )
    #' 
    #' # create 3d array: dim 1: samples; dim 2: time; dim 3: variables
    #' A <- array( c(A1,A2), dim=c(dim(A1),2) )
    #'     
    #' # predict
    #' B  <- predictr(model,
    #'                A[,dim(A)[2]:1,]     )
    #' B = B[,dim(B)[2]:1]
    #' # convert back to integers
    #' A1 <- bin2int(A1)
    #' A2 <- bin2int(A2)
    #' B  <- bin2int(B)
    #'  
    #' # inspect the differences              
    #' table( B-(A1+A2) )
    #' 
    #' # plot the difference
    #' hist(  B-(A1+A2) )
    #' }
    #' 
    predictr = function(model, X, hidden = FALSE, real_output = T,...){
      
      # coerce to array if matrix
      if(length(dim(X)) == 2){
        X <- array(X,dim=c(dim(X),1))
      }
      
      if(real_output && model$seq_to_seq_unsync){ ## here we modify the X in case of seq_2_seq & real_output to have the good dimensions
        time_dim_input = dim(X)[2]
        store = array(0, dim = c(dim(X)[1],model$time_dim,dim(X)[3]))
        store[,1:dim(X)[2],] = X
        X = store
        rm(store)
      }
      
      if(model$network_type == "rnn"){
        store = predict_rnn(model, X, hidden, real_output,...)
      } else if (model$network_type == "lstm"){
        store = predict_lstm(model, X, hidden, real_output,...)
      } else if (model$network_type == "gru"){
        store = predict_gru(model, X, hidden, real_output,...)
      }else{
        stop("network_type_unknown for the prediction")
      }
      
      if(real_output && model$seq_to_seq_unsync){
        if(length(dim(store)) == 2){
          store = store[,model$time_dim_input:model$time_dim,drop=F]
        }else{
          store = store[,model$time_dim_input:model$time_dim,,drop=F]
        }
      }
      
      return(store)
    }
    
    #' @name predict_rnn
    #' @importFrom stats runif
    #' @importFrom sigmoid sigmoid
    #' @title Recurrent Neural Network
    #' @description predict the output of a RNN model
    #' @param model output of the trainr function
    #' @param X array of input values, dim 1: samples, dim 2: time, dim 3: variables (could be 1 or more, if a matrix, will be coerce to array)
    #' @param hidden should the function output the hidden units states
    #' @param real_output option used when the function in called inside trainr, do not drop factor for 2 dimension array output
    #' @param ... arguments to pass on to sigmoid function
    #' @return array or matrix of predicted values
    
    predict_rnn <- function(model, X, hidden = FALSE, real_output = T,...) {
      
      store <- list()
      for(i in seq(length(model$synapse_dim) - 1)){
        store[[i]] <- array(0,dim = c(dim(X)[1:2],model$synapse_dim[i+1]))
      }
      
      # store the hidden layers values for each time step, needed in parallel of store because we need the t(-1) hidden states. otherwise, we could take the values from the store list
      layers_values  = list()
      for(i in seq(length(model$synapse_dim) - 2)){
        layers_values[[i]] <- matrix(0,nrow=dim(X)[1], ncol = model$synapse_dim[i+1])
      }
      
      for (position in 1:dim(X)[2]) {
        
        # generate input 
        x = array(X[,position,],dim=dim(X)[c(1,3)])
        
        for(i in seq(length(model$synapse_dim) - 1)){
          if (i == 1) { # first hidden layer, need to take x as input
            store[[i]][,position,] <- (x %*% model$time_synapse[[i]]) + (layers_values[[i]] %*% model$recurrent_synapse[[i]])
          } else if (i != length(model$synapse_dim) - 1 & i != 1){ #hidden layers not linked to input layer, depends of the last time step
            store[[i]][,position,] <- (store[[i-1]][,position,] %*% model$time_synapse[[i]]) + (layers_values[[i]] %*% model$recurrent_synapse[[i]])
          } else { # output layer depend only of the hidden layer of bellow
            store[[i]][,position,] <- store[[i-1]][,position,] %*% model$time_synapse[[i]]
          }
          if(model$use_bias){ # apply the bias if applicable
            store[[i]][,position,] <- store[[i]][,position,] + model$bias_synapse[[i]]
          }
          # apply the activation function
          store[[i]][,position,] <- sigmoid(store[[i]][,position,], method=model$sigmoid)
          
          if(i != length(model$synapse_dim) - 1){ # for all hidden layers, we need the previous state, looks like we duplicate the values here, it is also in the store list
            # store hidden layers so we can print it out. Needed for error calculation and weight iteration
            layers_values[[i]] = store[[i]][,position,]
          }
        }
      }
      
      # convert output to matrix if 2 dimensional, real_output argument added if used inside trainr
      if(real_output){
        if(dim(store[[length(store)]])[3]==1) {
          store[[length(store)]] <- matrix(store[[length(store)]],
                                           nrow = dim(store[[length(store)]])[1],
                                           ncol = dim(store[[length(store)]])[2])
        }
      }
      
      # return output
      if(hidden == FALSE){ # return only the last element of the list, i.e. the output
        return(store[[length(store)]])
      }else{ # return everything
        return(store)
      }
    }
    
    #' @name predict_lstm
    #' @importFrom stats runif
    #' @importFrom sigmoid sigmoid
    #' @title gru prediction function
    #' @description predict the output of a lstm model
    #' @param model output of the trainr function
    #' @param X array of input values, dim 1: samples, dim 2: time, dim 3: variables (could be 1 or more, if a matrix, will be coerce to array)
    #' @param hidden should the function output the hidden units states
    #' @param real_output option used when the function in called inside trainr, do not drop factor for 2 dimension array output
    #' @param ... arguments to pass on to sigmoid function
    #' @return array or matrix of predicted values
    
    predict_lstm <- function(model, X, hidden = FALSE, real_output = T,...) {
      
      store <- list()
      prev_layer_values = list()
      c_t = list()
      for(i in seq(length(model$hidden_dim))){
        store[[i]] = array(0,dim = c(dim(X)[1:2],model$hidden_dim[i],6)) # 4d arrays !!!, hidden, cell, f, i, g, o
        prev_layer_values[[i]]  = matrix(0,nrow=dim(X)[1], ncol = model$hidden_dim[i]) # we need this object because of t-1 which do not exist in store
        c_t[[i]]         = matrix(0,nrow=dim(X)[1], ncol = model$hidden_dim[i]) # we need this object because of t-1 which do not exist in store
      }
      store[[length(store)+1]] <- array(0,dim = c(dim(X)[1:2],model$output_dim))
      
      for (position in 1:dim(X)[2]) {
        
        # generate input
        x = array(X[,position,],dim=dim(X)[c(1,3)])
        
        for(i in seq(length(model$hidden_dim))){
          # hidden layer (input ~+ prev_hidden)
          f_t     = (x %*% array(model$time_synapse[[i]][,,1],dim=c(dim(model$time_synapse[[i]])[1:2]))) + (prev_layer_values[[i]] %*% array(model$recurrent_synapse[[i]][,,1],dim=c(dim(model$recurrent_synapse[[i]])[1:2]))) 
          i_t     = (x %*% array(model$time_synapse[[i]][,,2],dim=c(dim(model$time_synapse[[i]])[1:2]))) + (prev_layer_values[[i]] %*% array(model$recurrent_synapse[[i]][,,2],dim=c(dim(model$recurrent_synapse[[i]])[1:2]))) 
          c_in_t  = (x %*% array(model$time_synapse[[i]][,,3],dim=c(dim(model$time_synapse[[i]])[1:2]))) + (prev_layer_values[[i]] %*% array(model$recurrent_synapse[[i]][,,3],dim=c(dim(model$recurrent_synapse[[i]])[1:2]))) 
          o_t     = (x %*% array(model$time_synapse[[i]][,,4],dim=c(dim(model$time_synapse[[i]])[1:2]))) + (prev_layer_values[[i]] %*% array(model$recurrent_synapse[[i]][,,4],dim=c(dim(model$recurrent_synapse[[i]])[1:2])))
          if(model$use_bias){
            f_t = f_t + model$bias_synapse[[i]][,1]
            i_t = i_t + model$bias_synapse[[i]][,2]
            c_in_t = c_in_t + model$bias_synapse[[i]][,3]
            o_t = o_t + model$bias_synapse[[i]][,4]
          }
          f_t = sigmoid(f_t)
          i_t = sigmoid(i_t)
          c_in_t = tanh(c_in_t)
          o_t = sigmoid(o_t)
          
          c_t[[i]]     = f_t * c_t[[i]] + (i_t * c_in_t)
          store[[i]][,position,,1] = o_t * tanh(c_t[[i]])
          store[[i]][,position,,2] = c_t[[i]]
          store[[i]][,position,,3] = f_t
          store[[i]][,position,,4] = i_t
          store[[i]][,position,,5] = c_in_t
          store[[i]][,position,,6] = o_t
          
          # replace the x in case of multi layer
          prev_layer_values[[i]] = x = o_t * tanh(c_t[[i]])# the top of this layer at this position is the past of the top layer at the next position
        }
        
        
        # output layer (new binary representation)
        store[[length(store)]][,position,] = store[[length(store) - 1]][,position,,1] %*% model$time_synapse[[length(model$time_synapse)]]
        if(model$use_bias){
          store[[length(store)]][,position,] = store[[length(store)]][,position,] + model$bias_synapse[[length(model$bias_synapse)]]
        }
        store[[length(store)]][,position,] = sigmoid(store[[length(store)]][,position,])
      } # end time loop
      
      # convert output to matrix if 2 dimensional, real_output argument added if used inside trainr
      if(real_output){
        if(dim(store[[length(store)]])[3]==1) {
          store[[length(store)]] <- matrix(store[[length(store)]],
                                           nrow = dim(store[[length(store)]])[1],
                                           ncol = dim(store[[length(store)]])[2])
        }
      }
      
      # return output
      if(hidden == FALSE){ # return only the last element of the list, i.e. the output
        return(store[[length(store)]])
      }else{ # return everything
        return(store)
      }
    }
    
    
    
    #' @name predict_gru
    #' @importFrom stats runif
    #' @importFrom sigmoid sigmoid
    #' @title gru prediction function
    #' @description predict the output of a gru model
    #' @param model output of the trainr function
    #' @param X array of input values, dim 1: samples, dim 2: time, dim 3: variables (could be 1 or more, if a matrix, will be coerce to array)
    #' @param hidden should the function output the hidden units states
    #' @param real_output option used when the function in called inside trainr, do not drop factor for 2 dimension array output
    #' @param ... arguments to pass on to sigmoid function
    #' @return array or matrix of predicted values
    
    predict_gru <- function(model, X, hidden = FALSE, real_output = T,...) {
      
      store <- list()
      h_t = list()
      for(i in seq(length(model$hidden_dim))){
        store[[i]] = array(0,dim = c(dim(X)[1:2],model$hidden_dim[i],4)) # 4d arrays !!!, hidden, z, r, h
        h_t[[i]]         = matrix(0,nrow=dim(X)[1], ncol = model$hidden_dim[i]) # we need this object because of t-1 which do not exist in store
      }
      store[[length(store)+1]] <- array(0,dim = c(dim(X)[1:2],model$output_dim))
      
      for (position in 1:dim(X)[2]) {
        
        # generate input
        x = array(X[,position,],dim=dim(X)[c(1,3)])
        
        for(i in seq(length(model$hidden_dim))){
          # hidden layer (input ~+ prev_hidden)
          z_t     = (x %*% array(model$time_synapse[[i]][,,1],dim=c(dim(model$time_synapse[[i]])[1:2]))) + (h_t[[i]]  %*% array(model$recurrent_synapse[[i]][,,1],dim=c(dim(model$recurrent_synapse[[i]])[1:2]))) 
          r_t     = (x %*% array(model$time_synapse[[i]][,,2],dim=c(dim(model$time_synapse[[i]])[1:2]))) + (h_t[[i]]  %*% array(model$recurrent_synapse[[i]][,,2],dim=c(dim(model$recurrent_synapse[[i]])[1:2])))
          if(model$use_bias){
            z_t = z_t + model$bias_synapse[[i]][,1]
            r_t = r_t + model$bias_synapse[[i]][,2]
          }
          z_t = sigmoid(z_t)
          r_t = sigmoid(r_t)
          
          h_in_t  = (x %*% array(model$time_synapse[[i]][,,3],dim=c(dim(model$time_synapse[[i]])[1:2]))) + ((h_t[[i]]  * r_t) %*% array(model$recurrent_synapse[[i]][,,3],dim=c(dim(model$recurrent_synapse[[i]])[1:2])))
          if(model$use_bias){
            h_in_t = h_in_t + model$bias_synapse[[i]][,3]
          }
          h_in_t = tanh(h_in_t)
          
          h_t[[i]]     = (1 - z_t) * h_t[[i]] + (z_t * h_in_t)
          store[[i]][,position,,1] = h_t[[i]]
          store[[i]][,position,,2] = z_t
          store[[i]][,position,,3] = r_t
          store[[i]][,position,,4] = h_in_t
          
          # replace the x in case of multi layer
          x = h_t[[i]]  # the top of this layer at this position is the past of the top layer at the next position
        }
        
        
        # output layer (new binary representation)
        store[[length(store)]][,position,] = store[[length(store) - 1]][,position,,1] %*% model$time_synapse[[length(model$time_synapse)]]
        if(model$use_bias){
          store[[length(store)]][,position,] = store[[length(store)]][,position,] + model$bias_synapse[[length(model$bias_synapse)]]
        }
        store[[length(store)]][,position,] = sigmoid(store[[length(store)]][,position,])
      } # end time loop
      
      # convert output to matrix if 2 dimensional, real_output argument added if used inside trainr
      if(real_output){
        if(dim(store[[length(store)]])[3]==1) {
          store[[length(store)]] <- matrix(store[[length(store)]],
                                           nrow = dim(store[[length(store)]])[1],
                                           ncol = dim(store[[length(store)]])[2])
        }
      }
      
      # return output
      if(hidden == FALSE){ # return only the last element of the list, i.e. the output
        return(store[[length(store)]])
      }else{ # return everything
        return(store)
      }
    }
    
    
    
    ####
    
    error<-as.data.frame(c(1:epochD))
    colnames(error)<-"Iteration"
    
   rnn<- function (Y, X, model = NULL, learningrate, learningrate_decay = 1, 
              momentum = 0, hidden_dim = c(10), network_type = "rnn", numepochs = 1, 
              sigmoid = c("logistic", "Gompertz", "tanh"), use_bias = F, 
              batch_size = 1, seq_to_seq_unsync = F, update_rule = "sgd", 
              epoch_function = c(epoch_print, epoch_annealing), loss_function = loss_L1, 
              ...) 
    
    {
      sigmoid <- match.arg(sigmoid)
      if (length(dim(X)) == 2) {
        X <- array(X, dim = c(dim(X), 1))
      }
      if (length(dim(Y)) == 2) {
        Y <- array(Y, dim = c(dim(Y), 1))
      }
      if (seq_to_seq_unsync) {
        time_dim_input = dim(X)[2]
        store = array(0, dim = c(dim(X)[1], dim(X)[2] + dim(Y)[2] - 
                                   1, dim(X)[3]))
        store[, 1:dim(X)[2], ] = X
        X = store
        store = array(0, dim = c(dim(X)[1], time_dim_input + 
                                   dim(Y)[2] - 1, dim(Y)[3]))
        store[, time_dim_input:dim(store)[2], ] = Y
        Y = store
      }
      if (dim(X)[2] != dim(Y)[2] && !seq_to_seq_unsync) {
        stop("The time dimension of X is different from the time dimension of Y. seq_to_seq_unsync is set to FALSE")
      }
      if (dim(X)[1] != dim(Y)[1]) {
        stop("The sample dimension of X is different from the sample dimension of Y.")
      }
      if (is.null(model)) {
        model = list(...)
        model$input_dim = dim(X)[3]
        model$hidden_dim = hidden_dim
        model$output_dim = dim(Y)[3]
        model$synapse_dim = c(model$input_dim, model$hidden_dim, 
                              model$output_dim)
        model$time_dim = dim(X)[2]
        model$sigmoid = sigmoid
        model$network_type = network_type
        model$numepochs = numepochs
        model$batch_size = batch_size
        model$learningrate = learningrate
        model$learningrate_decay = learningrate_decay
        model$momentum = momentum
        model$update_rule = update_rule
        model$use_bias = use_bias
        model$seq_to_seq_unsync = seq_to_seq_unsync
        model$epoch_function = epoch_function
        model$loss_function = loss_function
        model$last_layer_error = Y * 0
        model$last_layer_delta = Y * 0
        if ("epoch_model_function" %in% names(model)) {
          stop("epoch_model_function is not used anymore, use epoch_function and return the model inside.")
        }
        if (seq_to_seq_unsync) {
          model$time_dim_input = time_dim_input
        }
        if (model$update_rule == "adagrad") {
          message("adagrad update, loss function not used and momentum set to 0")
          model$momentum = 0
        }
        model <- init_r(model)
        model$error <- array(0, dim = c(dim(Y)[1], model$numepochs))
      }
      else {
        message("retraining, all options except X, Y and the model itself are ignored, error are reseted")
        if (model$input_dim != dim(X)[3]) {
          stop("input dim changed")
        }
        if (model$time_dim != dim(X)[2]) {
          stop("time dim changed")
        }
        if (model$output_dim != dim(Y)[3]) {
          stop("output dim changed")
        }
        if (seq_to_seq_unsync && model$time_dim_input != time_dim_input) {
          stop("time input dim changed")
        }
        model$error <- array(0, dim = c(dim(Y)[1], model$numepochs))
      }
      for (epoch in seq(model$numepochs)) {
        error$error[epoch]<-colMeans(model$error)[model$current_epoch]
 
        incProgress(1/model$numepochs, detail = paste("Learning Rate:",
                                      lr, "Neurons in Recursive Neural Network:",nn, "Iteration:", model$current_epoch,
                                      "Error:", colMeans(model$error)[model$current_epoch] ))
        
        
        model$current_epoch = epoch
        index = sample(seq(round(dim(Y)[1]/model$batch_size)), 
                       dim(Y)[1], replace = T)
        lj = list()
        for (i in seq(round(dim(Y)[1]/model$batch_size))) {
          lj[[i]] = seq(dim(Y)[1])[index == i]
        }
        lj[unlist(lapply(lj, length)) < 1] = NULL
        for (j in lj) {
          a = X[j, , , drop = F]
          c = Y[j, , , drop = F]
          store = predictr(model, a, hidden = T, real_output = F)
          if (model$network_type == "rnn") {
            for (i in seq(length(model$synapse_dim) - 1)) {
              model$store[[i]][j, , ] = store[[i]]
            }
          }
          else if (model$network_type == "lstm" | model$network_type == 
                   "gru") {
            for (i in seq(length(model$hidden_dim))) {
              model$store[[i]][j, , , ] = store[[i]]
            }
            model$store[[length(model$hidden_dim) + 1]][j, 
                                                        , ] = store[[length(model$hidden_dim) + 1]]
          }
          model = backprop_r(model, a, c, j)
          if (model$update_rule == "sgd") {
            model = model$loss_function(model)
          }
          model = update_r(model)
        }
        for (i in model$epoch_function) {
          model <- i(model)
          if (!is.list(model)) {
            stop("one epoch function didn't return the model.")
          }
        }
      
        
        
      }
      if (colMeans(model$error)[epoch] <= min(colMeans(model$error)[1:epoch])) {
        model$store_best <- model$store
      }
      attr(model, "error") <- colMeans(model$error)
     
      
      return(model)
    }
    
   withProgress(message = 'Training Model:', value = 0, {
     
     
     
   model <- rnn(X = X.train,
                   Y = y.train,
                   learningrate = lr,
                  
                   sigmoid = c("Gompertz"),
                   numepochs = epochD,
                   hidden_dim = nn)
   
   })

   
   xtest <- df[,1:ncol(df),]
   xtest<- array(xtest, dim=c(1,ncol(df),28))
   predicted<-predictr(model, xtest)
   
   forplot<-as.data.frame(t(predicted))
   colnames(forplot)<-"predicted"
   
   forplot$real<-df[1,1:ncol(df),Pred]
   
   forplot$time<-c(1:nrow(forplot))
   forplot$date<-Values$Local.time
   
   Values$Predicted<-(t(predicted)*(max(Values$Open)-min(Values$Open)))+min(Values$Open)
      
   error$error<-model$error[1,]
   output$Plot <- renderPlot({

     P<-ggplot(forplot)+
  geom_line(aes(y=predicted, x=time ), colour="red")+
  geom_line(aes(y=real, x=time ), colour="blue")+
       xlim(L,ncol(forplot))+
       xlab("Time (10M)")+
       ylab("Relative Price (0-1)")+
       theme_minimal(base_size = 20)+
       ggtitle("Predicted Prices")+
       ylim(0,1)
       

print(P)
     
   })
   
   output$Plot2 <- renderPlot({
     
     Q<-ggplot(error)+
       geom_line(aes(y=error, x=Iteration ), colour="red")+

       
       xlab("Epoch")+
       ylab("Error")+
       theme_minimal(base_size = 20)+
       ggtitle("Error/Iteration")
     
     
     print(Q)
     
   })
   
   
   
   output$summary <- renderTable({
     Values
   })
      

  })
   
  output$help <- renderUI({
    
    HTML(paste(
      "","This application explores the possibilities of using a recursive neural network (RNN) to predict pseudo-stochastic time series.",
      "Forex (Foreign Exchange) is a highly dynamic and stochastic market which can be used to benchmark RNN", 
      "The application loads the price series and volumes from 7 different markets and use the interrelations among them to predict the price of currency 10 minutes after the last observation.",
      "Since the time-frame is 10 minutes the algorithm must analyze the last 3000 time points and find a pattern to predict the forthcoming values.",
      "","The application uses four parameters:","",
      "<b>Iterations:</b> Number of times that the dataset is presented to the algorithm. The higher the better but the slower. How new iterations improve the performance can be observed in the Error/Iteration plot",
      "","<b>Events Used for Training:</b> The number previous events are used to predict the coming events, the higher the better","",
      "<b>Neurons in the RNN:</b> RNN are special types of Neural Networks so it is possible to adjust the number of neurons in the deep layer, the effect of this number in the performance depends on the dataset","",
      "<b>Learning Rate:</b> This represents the size of the steps used to optimize the model. Low values are better but slower","",
      "","","This application is an implementation of the rnn package for R",
      sep = "</br>"
      
    ))
    
  })
  

}

# Run the application 
shinyApp(ui = ui, server = server)

