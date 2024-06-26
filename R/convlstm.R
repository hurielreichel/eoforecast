## A single step: convlstm_cell
# Source: Keyadana, 2020 (https://blogs.rstudio.com/ai/posts/2020-12-17-torch-convlstm/)

# Our convlstm_cell’s constructor takes arguments input_dim, hidden_dim, and bias,
# just like a torch LSTM Cell.But we’re processing two-dimensional input data.
# Instead of the usual affine combination of new input and previous state, we use a
# convolution of kernel size kernel_size. Inside convlstm_cell, it is self$conv that
# takes care of this.
#
# The channels dimension, which in the original input data would correspond to
# different variables, is creatively used to consolidate four convolutions into one:
# Each channel output will be passed to just one of the four cell gates.
# Once in possession of the convolution output, forward() applies the gate logic,
# resulting in the two types of states it needs to send back to the caller.

convlstm_cell <- nn_module(

  initialize = function(input_dim, hidden_dim, kernel_size, bias) {

    self$hidden_dim <- hidden_dim

    padding <- kernel_size %/% 2

    self$conv <- nn_conv2d(
      in_channels = input_dim + self$hidden_dim,
      # for each of input, forget, output, and cell gates
      out_channels = 4 * self$hidden_dim,
      kernel_size = kernel_size,
      padding = padding,
      bias = bias
    )
  },

  forward = function(x, prev_states) {

    h_prev <- prev_states[[1]]
    c_prev <- prev_states[[2]]

    combined <- torch_cat(list(x, h_prev), dim = 2)  # concatenate along channel axis
    combined_conv <- self$conv(combined)
    gate_convs <- torch_split(combined_conv, self$hidden_dim, dim = 2)
    cc_i <- gate_convs[[1]]
    cc_f <- gate_convs[[2]]
    cc_o <- gate_convs[[3]]
    cc_g <- gate_convs[[4]]

    # input, forget, output, and cell gates (corresponding to torch's LSTM)
    i <- torch_sigmoid(cc_i)
    f <- torch_sigmoid(cc_f)
    o <- torch_sigmoid(cc_o)
    g <- torch_tanh(cc_g)

    # cell state
    c_next <- f * c_prev + i * g
    # hidden state
    h_next <- o * torch_tanh(c_next)

    list(h_next, c_next)
  },

  init_hidden = function(batch_size, height, width) {

    list(
      torch_zeros(batch_size, self$hidden_dim, height, width, device = self$conv$weight$device),
      torch_zeros(batch_size, self$hidden_dim, height, width, device = self$conv$weight$device))
  }
)


## Iteration over time steps: convlstm

# A convlstm may consist of several layers, just like a torch LSTM. For each layer,
# we are able to specify hidden and kernel sizes individually.
#
# During initialization, each layer gets its own convlstm_cell. On call, convlstm
# executes two loops. The outer one iterates over layers. At the end of each iteration,
# we store the final pair (hidden state, cell state) for later reporting.
# The inner loop runs over input sequences, calling convlstm_cell at each time step.
#
# We also keep track of intermediate outputs, so we’ll be able to return the complete
# list of hidden_states seen during the process. Unlike a torch LSTM, we do this for
# every layer.

convlstm <- nn_module(

  # hidden_dims and kernel_sizes are vectors, with one element for each layer in n_layers
  initialize = function(input_dim, hidden_dims, kernel_sizes, n_layers, bias = TRUE) {

    self$n_layers <- n_layers

    self$cell_list <- nn_module_list()

    for (i in 1:n_layers) {
      cur_input_dim <- if (i == 1) input_dim else hidden_dims[i - 1]
      self$cell_list$append(convlstm_cell(cur_input_dim, hidden_dims[i], kernel_sizes[i], bias))
    }
  },

  # we always assume batch-first
  forward = function(x) {

    batch_size <- x$size()[1]
    seq_len <- x$size()[2]
    height <- x$size()[4]
    width <- x$size()[5]

    # initialize hidden states
    init_hidden <- vector(mode = "list", length = self$n_layers)
    for (i in 1:self$n_layers) {
      init_hidden[[i]] <- self$cell_list[[i]]$init_hidden(batch_size, height, width)
    }

    # list containing the outputs, of length seq_len, for each layer
    # this is the same as h, at each step in the sequence
    layer_output_list <- vector(mode = "list", length = self$n_layers)

    # list containing the last states (h, c) for each layer
    layer_state_list <- vector(mode = "list", length = self$n_layers)

    cur_layer_input <- x
    hidden_states <- init_hidden

    # loop over layers
    for (i in 1:self$n_layers) {

      # every layer's hidden state starts from 0 (non-stateful)
      h_c <- hidden_states[[i]]
      h <- h_c[[1]]
      c <- h_c[[2]]
      # outputs, of length seq_len, for this layer
      # equivalently, list of h states for each time step
      output_sequence <- vector(mode = "list", length = seq_len)

      # loop over timesteps
      for (t in 1:seq_len) {
        h_c <- self$cell_list[[i]](cur_layer_input[ , t, , , ], list(h, c))
        h <- h_c[[1]]
        c <- h_c[[2]]
        # keep track of output (h) for every timestep
        # h has dim (batch_size, hidden_size, height, width)
        output_sequence[[t]] <- h
      }

      # stack hs for all timesteps over seq_len dimension
      # stacked_outputs has dim (batch_size, seq_len, hidden_size, height, width)
      # same as input to forward (x)
      stacked_outputs <- torch_stack(output_sequence, dim = 2)

      # pass the list of outputs (hs) to next layer
      cur_layer_input <- stacked_outputs

      # keep track of list of outputs or this layer
      layer_output_list[[i]] <- stacked_outputs
      # keep track of last state for this layer
      layer_state_list[[i]] <- list(h, c)
    }

    list(layer_output_list, layer_state_list)

  }

)

## Tiny convlstm
train_convlstm <- function(train_dl,
                           val_dl,
                           num_epochs = 100,
                           plot_path = "vignettes/learning_curve.png",
                           input_dim = 1,
                           hidden_dims = c(64, 1),
                           kernel_sizes = c(3, 3),
                           n_layers = 2,
                           lr = 0.01,
                           .device){

  # model <- convlstm %>%
  #   setup(
  #     loss = nn_mse_loss(),
  #     optimizer = torch::optim_adam,
  #     ) %>%
  #   set_hparams(n_layers = n_layers,
  #               input_dim = input_dim,
  #               hidden_dims = hidden_dims,
  #               kernel_sizes = kernel_sizes)
  #
  #
  # rates_and_losses <- model %>% lr_finder(
  #   train_dl,
  #   # start_lr = 1e-4,
  #   # end_lr = 0.5,
  #   verbose = TRUE
  # )
  #
  # rates_and_losses %>% plot()

  # model %>% fit(train_dl, epochs = 100, valid_data = val_dl,
  #     callbacks = list(
  #       luz_callback_early_stopping(patience = 3),
  #       luz_callback_lr_scheduler(
  #         lr_one_cycle,
  #         max_lr = 0.01,
  #         epochs = 100,
  #         steps_per_epoch = length(train_dl),
  #         call_on = "on_batch_end")
  #     ),
  #     verbose = TRUE)
  #
    # lr_profiler <- luz_callback_record_lr(steps, verbose)

  #   fitted <- object %>%
  #     set_opt_hparams(lr = start_lr) %>%
  #     fit(...,
  #         data = data,
  #         epochs = 999999, # the callback will be responsible for interrupting
  #         callbacks = list(scheduler, lr_profiler),
  #         verbose = FALSE
  #     )
  #
  #   lr_records <- data.frame(sapply(fitted$records$lr_finder, as.numeric))
  #
  #   class(lr_records) <- c("lr_records", class(lr_records))
  #   lr_records
  # }

  model <- convlstm(input_dim = input_dim, hidden_dims = hidden_dims, kernel_sizes = kernel_sizes, n_layers = n_layers)
  model <- model$to(device = .device)

  ### Adam optimizer
  optimizer <- optim_adam(model$parameters, lr = lr)

  trn_loss <- c()
  val_loss <- c()
  epc <- c()
  ## Loop through Epochs
  cli::cli_progress_bar("Training convlstm", total = num_epochs)
  for (epoch in 1:num_epochs) {

    cli::cli_bullets(paste("\nEpoch:", epoch))
    cli::cli_rule()

    cli::cli_progress_update()
    model$train()
    batch_losses <- c()
    coro::loop(for (b in train_dl) {

      optimizer$zero_grad()

      # last-time-step output from last layer
      preds <- model(b$x)[[2]][[2]][[1]]

      # round(as.matrix(preds[1, 1, 1:91, 40:1]), 2) %>% image(main = "Prediction")
      # Sys.sleep(2)
      # round(as.matrix(b$y[1, 1, 1:91, 40:1]), 2) %>% image(main = "Validation")
      # Sys.sleep(2)
      # (round(as.matrix(preds[1, 1, 1:91, 40:1]), 2) - round(as.matrix(b$y[1, 1, 1:91, 40:1]), 2)) %>%
      #   image(main = "Error")

      loss <- nnf_mse_loss(preds, b$y)
      cli::cli_alert_info(paste("Training:", loss$item(), "\n"))

      batch_losses <- c(batch_losses, ifelse(is.infinite(loss$item()), NA, loss$item()))

      ### -------- Backpropagation --------
      loss$backward()
      ### -------- Update weights --------
      optimizer$step()

    })


    # Print Loss
    if (epoch %% 10 == 0){
      cat(sprintf("\nEpoch %d, training loss:%3f\n", epoch, mean(batch_losses, na.rm = T)))
      trn_loss <- c(trn_loss, mean(batch_losses, na.rm = T))
      epc <- c(epc, epoch)

      # # Early stopping
      try(print(trn_loss[epc]))
      # if (epoch >= 50 && mean(batch_losses, na.rm = TRUE) >= trn_loss[epoch - 40]) {
      #   cat("Validation loss did not improve. Early stopping...\n")
      #   break  # Stop training
      # } else {
      #   val_loss <- c(val_loss, mean(batch_losses, na.rm = TRUE))
      # }

    }

    # model$eval()
    # batch_losses <- c()
    #
    # # disable gradient tracking to reduce memory usage
    # with_no_grad({
    #   coro::loop(for (b in val_dl) {
    #
    #     # last-time-step output from last layer
    #     preds <- model(b$x)[[2]][[2]][[1]]
    #
    #     loss <- nnf_mse_loss(preds, b$y)
    #     cli::cli_alert_info(paste("Validation:", loss$item(), "\n"))
    #     batch_losses <- c(batch_losses, ifelse(is.infinite(loss$item()), NA, loss$item()))
    #   })
    # })
    #
    # # Print Loss
    # if (epoch %% 5 == 0){
    #   cat(sprintf("\nEpoch %d, validation loss:%3f\n", epoch, mean(batch_losses, na.rm = T)))
    #   val_loss <- c(val_loss, mean(batch_losses, na.rm = T))
    # }

  }

  cli::cli_progress_done()

  if (num_epochs >= 20){
    # create learning rate plot
    learning_curve <- tibble(epoch = epc, train_loss = trn_loss, validation_loss = val_loss) %>%
      tidyr::pivot_longer(cols = -epoch, names_to = "Dataset", values_to = "value") %>%
      ggplot(aes(x = epoch, y = value, col = Dataset)) +
      geom_line() +
      xlab("Epoch") +
      ylab("Loss") +
      ggtitle("Loss Curve")
    ggsave(plot_path, learning_curve)

  } else{
    cli::cli_alert_warning("No plotting, as num_epochs needs to be >= 20")
  }

  return(preds)

}

find_lr <- function(
                    train_dl,
                    plot_path = "vignettes/lr.png",
                    input_dim = 1,
                    hidden_dims = c(64, 1),
                    kernel_sizes = c(3, 3),
                    n_layers = 2,
                    .device,
                    min_lr = 0.0000001,
                    max_lr = 0.1,
                    steps = 5,
                    num_epochs = 4){

  model <- convlstm(input_dim = input_dim, hidden_dims = hidden_dims, kernel_sizes = kernel_sizes, n_layers = n_layers)
  model <- model$to(device = .device)

  Loss <- c()
  Learning_Rate <- c()

  lrs <- seq(min_lr, max_lr, length.out = steps)
  ## Loop through lrs
  for (i in lrs) {
    for (epoch in 1:num_epochs) {

      model$train()
      batch_losses <- c()
      coro::loop(for (b in train_dl) {

        optimizer <- optim_adam(model$parameters, lr = i)
        optimizer$zero_grad()

        # last-time-step output from last layer
        preds <- model(b$x)[[2]][[2]][[1]]

        loss <- nnf_mse_loss(preds, b$y)

        batch_losses <- c(batch_losses, ifelse(is.infinite(loss$item()), NA, loss$item()))

        ### -------- Backpropagation --------
        loss$backward()
        ### -------- Update weights --------
        optimizer$step()

      })

    }

    Loss <- c(Loss, mean(batch_losses, na.rm = T))
    Learning_Rate <- c(Learning_Rate, i)
    cli::cli_alert_info(paste("Learning Rate:", i, "Loss:", mean(batch_losses, na.rm = T), "\n"))
    gc()

    }

  lr_loss <- tibble(Loss, Learning_Rate) %>%
    filter(!is.infinite(Loss)) %>%
    ggplot(aes(x = Learning_Rate, y = Loss)) +
    geom_line() +
    ggtitle("Learning Rate Curve")
  ggsave(plot_path, lr_loss)

}


