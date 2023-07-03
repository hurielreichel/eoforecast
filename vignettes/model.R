## convLSTM
library(eoforecast)
# dummy input tensor
# x <- torch_rand(c(2, 4, 3, 16, 16)) # batch_size, seq_len, channels, height, width

dev <- torch_device(if(cuda_is_available()) {"cuda"}else{"cpu"})

## Sanity Check
# dl <- create_dummy_data()
# preds <- train_convlstm(train_dl = dl, val_dl = dl, 100, plot_path = "vignettes/learning_curve-sanity.png", .device = dev)

## Wrangle Cube to tensor
cube <- create_dl_from_cube(seq_len = 6, batch_size = 117)

## Find Optimal Learning Rate
cube[[1]] %>% find_lr(.device = dev,
                      input_dim = 1,
                      hidden_dims = c(64, 1),
                      kernel_sizes = c(3, 3),
                      n_layers = 2,
                      min_lr = 0.00000001,
                      max_lr = 0.0006,
                      steps = 10,
                      num_epochs = 3)

# Quick Training for sanity check
preds <- train_convlstm(cube[[1]], cube[[2]], num_epochs = 20, .device = dev, lr = 0.00001)

# Check Forecast
round(as.matrix(preds[50, 1, 1:40, 1:91]), 2) %>% image() ## dim = 80 1 40 91

# Train with Weather
weather_cube <- read_ecwmfr_netcdf()
cube_with_weather <- create_dl_with_weather()
preds <- train_convlstm(cube_with_weather, num_epochs = 6)

preds %>% saveRDS("vignettes/preds.rds")
