read_ecwmfr_netcdf <- function(path_to_nc = "vignettes/data/adaptor.mars.internal-1685729229.7289045-5910-16-183eef6f-2e18-4e4b-a71c-c43248c6cb78.nc",
                               path_to_cube = "vignettes/data/switzerland/openEO_2019-01-03Z.tif"){

  # read base s5p data
  base_cube <- read_stars(path_to_cube)
  st_crs(base_cube) <- crs(4326)

  # Read the NetCDF file
  weather_cube <- stars::read_stars(path_to_nc) %>%
    # dplyr::select(u10) %>%
    aggregate(by = "day", FUN = mean) %>%
    # st_transform(., base_cube) %>%
    st_warp(., base_cube)

  return(weather_cube)

}

create_dl_with_weather <- function(weather_cube, path_to_tiffs = "vignettes/data/switzerland", batch_size = 100, train_pct = 0.8, seq_len = 4) {

  # Get all the file paths in the directory
  file_paths <- list.files(path = path_to_tiffs, pattern = "*\\.tif$", full.names = TRUE)

  dates <- gsub(".*_(\\d{4}-\\d{2}-\\d{2})Z\\.tif", "\\1", file_paths) %>% as.Date()
  weather_cube <- weather_cube %>% dplyr::filter(lubridate::date(time) %in% dates) %>%
    mutate(u10 = 100 * ((u10 - min(u10)) / (max(u10) - min(u10))),
           t2m = 100 * ((t2m - min(t2m)) / (max(t2m) - min(t2m))),
           tp = 100 * ((tp - min(tp)) / (max(tp) - min(tp))),
           weather = u10 * t2m + tp
           ) %>%
    dplyr::select(weather)
    # plot(weather_cube, names = NULL, col = grey((5:10)/10))

  # Read the files and combine into a single array
  cube_array <- array(0, dim = c(length(file_paths), length(file_paths), 1, nrow(raster::raster(file_paths[[1]])), ncol(raster::raster(file_paths[[1]]))))
  for (i in seq_along(file_paths)) {
    cube_array[i,,,,] <- raster(file_paths[[i]])[]
    cube_array[,i,,,] <- weather_cube[,,,i]$weather %>% as.array()
  }

  # Create sequence tensor
  cube_array_seq <- array(0, dim = c(dim(cube_array)[1] - seq_len + 1, dim(cube_array)[2], seq_len, dim(cube_array)[4], dim(cube_array)[5]))
  for (i in 1:(dim(cube_array)[1] - seq_len + 1)) {
    cube_array_seq[i,,,,] <- cube_array[i:(i + seq_len - 1),,,,]
  }

  rm(cube_array)
  gc()

  # Replace NAs and NaNs with 0
  cube_array_seq[is.na(cube_array_seq)] <- 0
  cube_array_seq[is.nan(cube_array_seq)] <- 0

  # convert to tensor
  cube_tensor <- torch_tensor(cube_array_seq, dtype = torch_float())

  rm(cube_array_seq)
  gc()

  # Convert cube_tensor to data frame
  # df <- reshape2::melt(cube_tensor %>% as.array(), varnames = c("Time", "Seq", "Channel", "X", "Y"))
  #
  # # Rename columns
  # names(df) <- c("Time", "Seq", "Channel", "X", "Y", "Value")
  # df2 <- df %>% filter(Time == 200)
  #
  # ggplot(df2, aes(x = X, y = Y, fill = Value)) +
  #   geom_raster() +
  #   scale_fill_gradient(low = "white", high = "red")

  # Split into training and validation sets
  train_size <- floor(seq_len * train_pct)
  # train_data <- cube_tensor[1:train_size, , , , ]
  # val_data <- cube_tensor[(train_size+1):dim(cube_tensor)[1], , , , ]

  full_size <- dim(cube_tensor)[2]

  dummy_ds <- dataset(

    initialize = function(data) {
      self$data <- data
    },

    .getitem = function(i) {
      list(x = self$data[i, 1:train_size, ..], y = self$data[i, seq_len, ..])
    },

    .length = function() {
      nrow(self$data)
    }
  )

  # Create dataloaders from the datasets
  train_ds <- dummy_ds(cube_tensor)
  train_dl <- dataloader(train_ds, batch_size = batch_size, shuffle = FALSE)

  cli_alert_info(paste("Validation Set start at", train_size+1))
  return(train_dl)
}

