## convLSTM
library(eoforecast)
# dummy input tensor
# x <- torch_rand(c(2, 4, 3, 16, 16)) # batch_size, seq_len, channels, height, width

device <- torch_device(if(cuda_is_available()) {"cuda"}else{"cpu"})

## Sanity Check
# dl <- create_dummy_data()
# preds <- train_convlstm(dl = dl, 10, plot_path = "vignettes/learning_curve-sanity.png")

## Wrangle Cube to tensor
cube <- create_dl_from_cube()

# Quick Training for sanity check
preds <- train_convlstm(cube, num_epochs = 6)

# Check Forecast
round(as.matrix(preds[1, 1, 1:40, 1:91]), 2) %>% image() ## dim = 80 1 40 91

# Train with Weather
weather_cube <- read_ecwmfr_netcdf()
cube_with_weather <- create_dl_with_weather()
preds <- train_convlstm(cube_with_weather, num_epochs = 6)
