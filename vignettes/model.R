## convLSTM
library(eoforecast)
library(ggplot2)
library(reshape2)
library(spatstat.explore)

# dummy input tensor
# x <- torch_rand(c(2, 4, 3, 16, 16)) # batch_size, seq_len, channels, height, width

dev <- torch_device(if(cuda_is_available()) {"cuda"}else{"cpu"})

## Sanity Check
# dl <- create_dummy_data()
# preds <- train_convlstm(train_dl = dl, val_dl = dl, 200, plot_path = "vignettes/learning_curve-sanity2.png", .device = dev,
#                         input_dim = 1,
#                         hidden_dims = c(64, 1),
#                         kernel_sizes = c(3, 3),
#                         n_layers = 2)

################################################################################
#      _____ __________
#     / ___// ____/ __ \
#     \__ \/___ \/ /_/ /
#   ___ / /___/ / ____/
#   /____/_____/_/
################################################################################


## Wrangle Cube to tensor
# Get all the file paths in the directory
file_paths <- list.files(path = "vignettes/data/switzerland",
                         pattern = "*\\.tif$", full.names = TRUE)

# Find the maximum value
max_v = read_stars(file_paths) %>% merge(f = max) %>% pull() %>% max(na.rm = T)
window_size = 30

# Check if window_size is valid
if (window_size > length(file_paths)) {
  stop("Window size is larger than the number of files available.")
}

# Create an initial sequence the size of the window
cube_window <- vector("list", window_size)
for (i in 1:window_size){
  idx = length(file_paths)-window_size+i
  no2_data <- read_stars(file_paths[[idx]]) %>% pull()
  no2_data <- no2_data[,40:1]
  # no2_data[no2_data<0] <- 0
  # no2_data[no2_data>100] <- 100 # rough outlier removal
  no2_data <- no2_data %>% as.im() %>% blur(3) %>% as.matrix()
  cube_window[[i]] <- no2_data / max_v # scaling from 0 to 1
  # cube_window[[i]][is.na(cube_window[[i]])] <- 0
  # cube_window[[i]][is.nan(cube_window[[i]])] <- 0
  # cube_window[[i]][cube_window[[i]]<0] <- 0
  # cube_window[[i]][cube_window[[i]]>100] <- 0 # rough outlier removal
  cube_window[[i]] <- cube_window[[i]] %>% torch_tensor()
}
init_seq <- torch_stack(cube_window, dim = 1)

# Read the files and combine into a single array
cube_rst <- vector("list", length(file_paths)-1)
for (i in seq_along(file_paths[1:length(file_paths)-1])) {
  no2_data <- read_stars(file_paths[[i]]) %>% pull()
  no2_data <- no2_data[,40:1] %>% as.im() %>% blur(3) %>% as.matrix()
  # no2_data[no2_data<0] <- 0
  # no2_data[is.na(no2_data)] <- 0
  # no2_data[no2_data>100] <- 100 # rough outlier removal
  no2_data <- no2_data / max_v

  cube_rst[[i]] <- no2_data  %>% torch_tensor()
}

# The data may not have a pair length, which complicates for the windows
# For this reason, we'll compute the remainder and not use the beginning of the
# time series, but only the "end"
beamed_size <-  as.integer(length(cube_rst) / window_size)
remainder <- length(cube_rst) %% beamed_size + 1
cube_rst <- cube_rst[1:(length(cube_rst)-remainder)]

cube_beams <- vector("list", beamed_size)
cube_beams[[1]] <- init_seq # beginning of our windows data will be forced
for (i in 2:beamed_size){
  from = (i - 1) * window_size + 1
  to = from + window_size - 1
  cube_beams[[i]] <- cube_rst[from:to] %>% torch_stack(., 1)
}

input <- torch_stack(cube_beams, dim = 2)
input <- input$unsqueeze(3) # create artificial dimension to squeeze into input format
input %>% dim()

rm(cube_rst, cube_beams)
gc()

data_size = dim(input)[2]
trn_size = data_size - 1

create_train_ds <- dataset(

  initialize = function(data) {
    self$data <- data
  },

  .getitem = function(i) {
    list(x = self$data[i, 1:trn_size, ..], y = self$data[i, data_size, ..])
  },

  .length = function() {
    nrow(self$data)
  }
)

# Create dataloaders from the datasets
train_ds <- create_train_ds(input)
rm(input)
gc()
train_dl <- dataloader(train_ds, batch_size = window_size)

## Find Optimal Learning Rate
train_dl %>% find_lr(.device = dev,
                      input_dim = 1,
                      hidden_dims = c(64, 1),
                      kernel_sizes = c(3, 3),
                      n_layers = 2,
                      min_lr = 0.00001,
                      max_lr = 0.01,
                      steps = 60,
                      num_epochs = 2)

gc()

# Quick Training for sanity check
preds <- train_convlstm(train_dl, train_dl, num_epochs = 200,
                        .device = dev,
                        lr = 0.001, # 0.0001
                        input_dim = 1,
                        hidden_dims = c(64, 1),
                        kernel_sizes = c(3, 3),
                        n_layers = 2
                        )

library(gridExtra)
real_plots <- list()
pred_plots <- list()
error_plots <- list()
for (i in 1:30){
  # Check Forecast
  preds[i,1,,] %>% as.matrix() %>% image()

  prd <- preds[i,1,,] %>% as.matrix()
  val <- cube_window[[i]] %>% as.matrix()

  # plot the prediction
  # Reshape the matrix for ggplot
  prd_df <- melt(prd)

  prd_plt <- ggplot(prd_df, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile() +
    scale_fill_viridis_c() +
    labs(x = "X", y = "Y", title = "NO2 0-1 Forecast") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    guides(fill = guide_colorbar(title = "NO2 Prediction"))
  pred_plots[[i]] <- prd_plt

  # plot the real value
  # Reshape the matrix for ggplot
  prd_df <- melt(val)

  real_plt <- ggplot(prd_df, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile() +
    scale_fill_viridis_c() +
    labs(x = "X", y = "Y", title = "NO2 0-1") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    guides(fill = guide_colorbar(title = "NO2 Real Value"))
  real_plots[[i]] <- real_plt


  # plot the error
  # Reshape the matrix for ggplot
  mat_df <- melt(prd-val)
  rmse = (mat_df$value ** 2) %>% sqrt() %>% mean() %>% round(2)
  error_noweather <- c(error_noweather, rmse)

  error_plt <- ggplot(mat_df, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", mid = "white", midpoint = 0, high = "red", limits = c(-.5, .5)) +
    labs(x = "X", y = "Y", title = paste("Error Map - RMSE=", as.character(rmse))) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    guides(fill = guide_colorbar(title = "Error"))
  error_plots[[i]] <- error_plt
}


  # Check Forecast
preds[30,1,,] %>% as.matrix() %>% image()

prd <- preds[30,1,,] %>% as.matrix()
val <- cube_window[[30]] %>% as.matrix()

# plot the prediction
# Reshape the matrix for ggplot
prd_df <- melt(prd)

ggplot(prd_df, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_viridis_c() +
  labs(x = "X", y = "Y", title = "NO2 0-1 Forecast") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  guides(fill = guide_colorbar(title = "NO2 Prediction"))

# plot the real value
# Reshape the matrix for ggplot
prd_df <- melt(val)

ggplot(prd_df, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_viridis_c() +
  labs(x = "X", y = "Y", title = "NO2 0-1") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  guides(fill = guide_colorbar(title = "NO2 Real Value"))

# plot the error
# Reshape the matrix for ggplot
mat_df <- melt(prd-val)
rmse = (mat_df$value ** 2) %>% sqrt() %>% mean() %>% round(2)

# Create a ggplot object
ggplot(mat_df, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", mid = "white", midpoint = 0, high = "red", limits = c(-.5, .5)) +
  labs(x = "X", y = "Y", title = paste("Error Map - RMSE=", as.character(rmse))) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  guides(fill = guide_colorbar(title = "Error"))

################################################################################
#    _       __           __  __
#   | |     / /__  ____ _/ /_/ /_  ___  _____
#   | | /| / / _ \/ __ `/ __/ __ \/ _ \/ ___/
#   | |/ |/ /  __/ /_/ / /_/ / / /  __/ /
#   |__/|__/\___/\__,_/\__/_/ /_/\___/_/
################################################################################

# Train with Weather
weather_cube <- read_ecwmfr_netcdf()

# Create tensor once more
dates <- gsub(".*_(\\d{4}-\\d{2}-\\d{2})Z\\.tif", "\\1", file_paths) %>% as.Date()
weather_cube <- weather_cube %>% dplyr::filter(lubridate::date(time) %in% dates) %>%
  mutate(u10 = (u10 - min(u10)) / (max(u10) - min(u10)),
         t2m = (t2m - min(t2m)) / (max(t2m) - min(t2m)),
         tp = (tp - min(tp)) / (max(tp) - min(tp)),
         weather = u10 * t2m + tp
  ) %>%
  dplyr::select(weather)
# plot(weather_cube, names = NULL, col = grey((5:10)/10))

# get the maximum value
max_weather <- weather_cube %>% pull() %>% max()

# Check if window_size is valid
if (window_size > length(file_paths)) {
  stop("Window size is larger than the number of files available.")
}

# Create an initial sequence the size of the window
cube_array <- array(0, dim = c(window_size, 2, ncol(raster::raster(file_paths[[1]])), nrow(raster::raster(file_paths[[1]]))))
cube_window <- vector("list", window_size)
for (i in 1:window_size){

  idx = length(file_paths)-window_size+i
  no2_data <- read_stars(file_paths[[idx]]) %>% pull()
  no2_data <- no2_data[,40:1] %>% as.im() %>% blur(3) %>% as.matrix()
  # no2_data[no2_data<0] <- 0
  # no2_data[no2_data>100] <- 100 # rough outlier removal
  no2_data <- no2_data / max_v

  weather_data <- weather_cube %>%
    slice(index = length(file_paths)-window_size+i, along = "time") %>%
    pull()
  weather_data <- weather_data[,40:1] / max_weather

  cube_array[i,1,,] <- no2_data
  cube_array[i,2,,] <- weather_data
  cube_window[[i]] <- cube_array[i,,,]
  # cube_window[[i]][is.na(cube_window[[i]])] <- 0
  # cube_window[[i]][is.nan(cube_window[[i]])] <- 0
  cube_window[[i]] <- cube_window[[i]] %>% torch_tensor()
}
init_seq <- torch_stack(cube_window, dim = 1)

# a quick look into what we're validating with
cube_window[[30]][1,,] %>% as.matrix() %>% image()
cube_window[[30]][2,,] %>% as.matrix() %>% image()

# Read the files and combine into a single array
cube_array <- array(0, dim = c( length(file_paths)-1, 2, ncol(raster::raster(file_paths[[1]])), nrow(raster::raster(file_paths[[1]]))))
cube_rst <- vector("list", length(file_paths)-1)
for (i in seq_along(file_paths[1:length(file_paths)-1])){

  no2_data <- read_stars(file_paths[[i]]) %>% pull()
  no2_data <- no2_data[,40:1] %>% as.im() %>% blur(3) %>% as.matrix()
  # no2_data[no2_data<0] <- 0
  # no2_data[is.na(no2_data)] <- 0
  # no2_data[no2_data>100] <- 100 # rough outlier removal
  no2_data <- no2_data / max_v

  weather_data <- weather_cube %>% slice(index = i, along = "time") %>% pull()
  weather_data <- weather_data[,40:1]

  cube_array[i,1,,] <- no2_data
  cube_array[i,2,,] <- weather_data
  cube_rst[[i]] <- cube_array[i,,,]
  # cube_rst[[i]][is.na(cube_rst[[i]])] <- 0
  # cube_rst[[i]][is.nan(cube_rst[[i]])] <- 0
  cube_rst[[i]] <- cube_rst[[i]] %>% torch_tensor()
}

# The data may not have a pair length, which complicates for the windows
# For this reason, we'll compute the remainder and not use the beginning of the
# time series, but only the "end"
beamed_size <-  as.integer(length(cube_rst) / window_size)
remainder <- length(cube_rst) %% beamed_size + 1
cube_rst <- cube_rst[1:(length(cube_rst)-remainder)]

cube_beams <- vector("list", beamed_size)
cube_beams[[1]] <- init_seq # beginning of our windows data will be forced
for (i in 2:beamed_size){
  from = (i - 1) * window_size + 1
  to = from + window_size - 1 # i am still ignoring 361
  cube_beams[[i]] <- cube_rst[from:to] %>% torch_stack(., 1)
}

input <- torch_stack(cube_beams, dim = 2)
# input <- input$unsqueeze(3) # create artificial dimension to squeeze into input format
input %>% dim()

rm(cube_rst, cube_beams)
gc()

data_size = dim(input)[2]
trn_size = data_size - 1

# Create dataloaders from the datasets
train_ds <- create_train_ds(input)
rm(input)
gc()
train_dl <- dataloader(train_ds, batch_size = window_size)

preds_wet <- train_convlstm(train_dl, train_dl, num_epochs = 200,
                        .device = dev,
                        lr = 0.001, # 0.0001
                        input_dim = 2,
                        hidden_dims = c(64, 1),
                        kernel_sizes = c(3, 3),
                        n_layers = 2,
)

# Check Forecast

errors <- c()
for (i in 1:30){

  # preds_wet[i,1,,] %>% as.matrix() %>% image()
  prd <- preds_wet[i,1,,] %>% as.matrix()
  val <- cube_window[[i]][1,,] %>% as.matrix()

  # plot the prediction
  # Reshape the matrix for ggplot
  prd_df <- melt(prd)

  prd_plt <- ggplot(prd_df, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile() +
    scale_fill_viridis_c() +
    labs(x = "X", y = "Y", title = "NO2 0-1") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    guides(fill = guide_colorbar(title = "NO2 Prediction"))
  pred_plots[[i]] <- prd_plt

  # plot the error
  # Reshape the matrix for ggplot
  mat_df <- melt(prd-val)
  rmse = (mat_df$value ** 2) %>% sqrt() %>% mean() %>% round(2)
  print(rmse)
  # errors <- c(errors, rmse)
  # plot the error
  # Reshape the matrix for ggplot
  mat_df <- melt(prd-val)
  rmse = (mat_df$value ** 2) %>% sqrt() %>% mean() %>% round(2)
  # print(rmse)
  # error_noweather <- c(error_noweather, rmse)
  error_plt <- ggplot(mat_df, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", mid = "white", midpoint = 0, high = "red", limits = c(-.5, .5)) +
    labs(x = "X", y = "Y", title = paste("Error Map - RMSE=", as.character(rmse))) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    guides(fill = guide_colorbar(title = "Error"))
  error_plots[[i]] <- error_plt
}

preds_wet[30,1,,] %>% as.matrix() %>% image()
prd <- preds_wet[30,1,,] %>% as.matrix()
val <- cube_window[[30]][1,,] %>% as.matrix()

# plot the prediction
# Reshape the matrix for ggplot
prd_df <- melt(prd)

ggplot(prd_df, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_viridis_c(limits = c(0,.4)) +
  labs(x = "X", y = "Y", title = "NO2 0-1") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  guides(fill = guide_colorbar(title = "NO2 Prediction"))

# plot the real value
# Reshape the matrix for ggplot
prd_df <- melt(val)

ggplot(prd_df, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_viridis_c(limits = c(0,.4)) +
  labs(x = "X", y = "Y", title = "NO2 0-1") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  guides(fill = guide_colorbar(title = "NO2 Real Value"))

# plot the error
# Reshape the matrix for ggplot
mat_df <- melt(prd-val)
rmse = (mat_df$value ** 2) %>% sqrt() %>% mean() %>% round(2)
cat(rmse)

# Create a ggplot object
ggplot(mat_df, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", mid = "white", midpoint = 0, high = "red", limits = c(-.5, .5)) +
  labs(x = "X", y = "Y", title = paste("Error Map - RMSE=", as.character(rmse))) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  guides(fill = guide_colorbar(title = "Error"))

no2 <- c()
weather <- c()
for (i in 1:30){
  no2 <- c(no2, cube_window[[i]][1,,] %>% as.matrix())
  weather <- c(weather, cube_window[[i]][2,,] %>% as.matrix())
}

tibble(NO2 = no2, Weather = weather) %>%
  ggplot(aes(x = Weather, y = NO2)) + geom_point(alpha=0.1, size = 0.5) +
  labs(title = "Relationship between weather feature and NO2 data")



  weather <- cube_window[[24]][2,,] %>% as.matrix() %>% melt()
  ggplot(weather, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile() +
    scale_fill_viridis_c(limits = c(0,0.4)) +
    labs(x = "X", y = "Y", title = "Weather Featured Data") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    guides(fill = guide_colorbar(title = "Weather 0 - 1"))
