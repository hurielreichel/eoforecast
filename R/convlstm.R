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

    c(h_prev, c_prev) %<-% prev_states

    combined <- torch_cat(list(x, h_prev), dim = 2)  # concatenate along channel axis
    combined_conv <- self$conv(combined)
    c(cc_i, cc_f, cc_o, cc_g) %<-% torch_split(combined_conv, self$hidden_dim, dim = 2)

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

    c(batch_size, seq_len, num_channels, height, width) %<-% x$size()

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
      c(h, c) %<-% hidden_states[[i]]
      # outputs, of length seq_len, for this layer
      # equivalently, list of h states for each time step
      output_sequence <- vector(mode = "list", length = seq_len)

      # loop over time steps
      for (t in 1:seq_len) {
        c(h, c) %<-% self$cell_list[[i]](cur_layer_input[ , t, , , ], list(h, c))
        # keep track of output (h) for every time step
        # h has dim (batch_size, hidden_size, height, width)
        output_sequence[[t]] <- h
      }

      # stack hs for all time steps over seq_len dimension
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
train_convlstm <- function(dl,
                           num_epochs = 100,
                           plot_path = "vignettes/learning_curve.png",
                           input_dim = 1,
                           hidden_dims = c(64, 1),
                           kernel_sizes = c(3, 3),
                           n_layers = 2,
                           .device = device){

  model <- convlstm(input_dim = input_dim, hidden_dims = hidden_dims, kernel_sizes = kernel_sizes, n_layers = n_layers)
  model <- model$to(device = .device)

  ### Adam optimizer
  optimizer <- optim_adam(model$parameters)

  losses <- c()
  ## Loop through Epochs
  cli::cli_progress_bar("Training convlstm", total = num_epochs)
  for (epoch in 1:num_epochs) {

    cli::cli_progress_update()
    model$train()
    batch_losses <- c()

    coro::loop(for (b in dl) {

      optimizer$zero_grad()

      # last-time-step output from last layer
      preds <- model(b$x)[[2]][[2]][[1]]

      loss <- nnf_mse_loss(preds, b$y)
      losses <- c(losses, as.numeric(loss))
      cli::cli_alert(paste("Loss:", (loss %>% as.numeric() %>% round(3))))

      loss$backward()
      optimizer$step()

      train_losses <- c(train_losses, loss$item)

    })

    # Early stopping
    if (epoch > 10){
      if (losses[epoch] >= losses[1]){
        stop(cli::cli_abort("Early Stopping triggered"))
      }
    }

    if (epoch %% 10 == 0)
      cat(sprintf("\nEpoch %d, training loss:%3f\n", epoch, mean(batch_losses, na.rm = T)))

    model$eval()

  }

  cli::cli_progress_done()

  losses <- zoo::rollapply(losses, width = 3, FUN = mean, by = 3, align = "left", fill = NA) %>% na.omit()

  # create learning rate plot
  learning_curve <- tibble(epoch = 1:num_epochs, loss = losses) %>%
    ggplot(aes(x = epoch, y = loss)) +
    geom_line() +
    xlab("Epochs") +
    ylab("Loss") +
    ggtitle("Learning Rate Curve")
  ggsave(plot_path, learning_curve)

  return(preds)

}
