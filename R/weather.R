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

create_dl_with_weather <- function(weather_cube, path_to_tiffs = "vignettes/data/switzerland", window_size = 30) {

  # Get all the file paths in the directory
  file_paths <- list.files(path = path_to_tiffs, pattern = "*\\.tif$", full.names = TRUE)

  dates <- gsub(".*_(\\d{4}-\\d{2}-\\d{2})Z\\.tif", "\\1", file_paths) %>% as.Date()
  weather_cube <- weather_cube %>% dplyr::filter(lubridate::date(time) %in% dates) %>%
    mutate(u10 = (u10 - min(u10)) / (max(u10) - min(u10)),
           t2m = (t2m - min(t2m)) / (max(t2m) - min(t2m)),
           tp = (tp - min(tp)) / (max(tp) - min(tp)),
           weather = u10 * t2m + tp
           ) %>%
    dplyr::select(weather)
    # plot(weather_cube, names = NULL, col = grey((5:10)/10))

  # Read the files and combine into a single array
  # cube_array <- array(0, dim = c(length(file_paths), length(file_paths), 1, ncol(raster::raster(file_paths[[1]])), nrow(raster::raster(file_paths[[1]]))))
  # for (i in seq_along(file_paths)) {
  #   cube_array[i,,,,] <- raster(file_paths[[i]])[]
  #   cube_array[,i,,,] <- weather_cube[,,,i]$weather %>% as.array()
  # }

  # Check if window_size is valid
  if (window_size > length(file_paths)) {
    stop("Window size is larger than the number of files available.")
  }

  # Create an initial sequence the size of the window
  cube_array <- array(0, dim = c(window_size, 2, ncol(raster::raster(file_paths[[1]])), nrow(raster::raster(file_paths[[1]]))))
  cube_window <- vector("list", window_size)
  for (i in 1:window_size){

    no2_data <- read_stars(file_paths[[length(file_paths)-window_size+i]]) %>% pull()
    no2_data <- no2_data[,40:1]
    no2_data[no2_data<0] <- 0
    no2_data[no2_data>100] <- 100 # rough outlier removal
    no2_data <- no2_data / 100

    weather_data <- weather_cube %>%
      slice(index = length(file_paths)-window_size+i, along = "time") %>%
      pull()
    weather_data <- weather_data[,40:1]

    cube_array[i,1,,] <- no2_data
    cube_array[i,2,,] <- weather_data
    cube_window[[i]] <- cube_array[i,,,]
    cube_window[[i]][is.na(cube_window[[i]])] <- 0
    cube_window[[i]][is.nan(cube_window[[i]])] <- 0
    cube_window[[i]] <- cube_window[[i]] %>% torch_tensor()
  }
  init_seq <- torch_stack(cube_window, dim = 1)

  # Read the files and combine into a single array
  cube_array <- array(0, dim = c( length(file_paths), 2, ncol(raster::raster(file_paths[[1]])), nrow(raster::raster(file_paths[[1]]))))
  cube_rst <- vector("list", length(file_paths))
  for (i in seq_along(file_paths)){

    no2_data <- read_stars(file_paths[[i]]) %>% pull()
    no2_data <- no2_data[,40:1]
    no2_data[no2_data<0] <- 0
    no2_data[no2_data>100] <- 100 # rough outlier removal
    no2_data <- no2_data / 100

    weather_data <- weather_cube %>% slice(index = i, along = "time") %>% pull()
    weather_data <- weather_data[,40:1]

    cube_array[i,1,,] <- no2_data
    cube_array[i,2,,] <- weather_data
    cube_rst[[i]] <- cube_array[i,,,]
    cube_rst[[i]][is.na(cube_rst[[i]])] <- 0
    cube_rst[[i]][is.nan(cube_rst[[i]])] <- 0
    cube_rst[[i]] <- cube_rst[[i]] %>% torch_tensor()
  }

  # The data may not have a pair length, which complicates for the windows
  # For this reason, we'll compute the remainder and not use the beginning of the
  # time series, but only the "end"
  beamed_size <-  as.integer(length(cube_rst) / window_size)
  remainder <- length(cube_rst) %% beamed_size + 1
  cube_rst <- cube_rst[remainder:length(cube_rst)]

  cube_beams <- vector("list", beamed_size)
  cube_beams[[1]] <- init_seq # beginning of our windows data will be forced
  for (i in 2:beamed_size){
    from = (i - 1) * window_size + 1
    to = from + window_size - 1 # i am still ignoring 361
    cube_beams[[i]] <- cube_rst[from:to] %>% torch_stack(., 1)
  }

  input <- torch_stack(cube_beams, dim = 1)
  # input <- input$unsqueeze(3) # create artificial dimension to squeeze into input format
  input %>% dim()

  rm(cube_rst, cube_window, cube_beams)
  gc()

  # Replace NAs and NaNs with 0
  # cube_array_seq[is.na(cube_array_seq)] <- 0
  # cube_array_seq[is.nan(cube_array_seq)] <- 0
  # cube_array_seq[cube_array_seq<0] <- 0
  # cube_array_seq[cube_array_seq>100] <- 0 # rough outlier removal
  #
  # # Calculate minimum and maximum values
  # min_val <- min(cube_array_seq, na.rm = TRUE)
  # max_val <- max(cube_array_seq, na.rm = TRUE)
  #
  # cli::cli_alert_info(paste("Min Value = ", min_val, "; Max Value =", max_val))
  #
  # # Shift the values to have a minimum of 0
  # shifted_array <- cube_array_seq - min_val
  #
  # # Scale the values between 0 and 1
  # cube_array_seq <- shifted_array / (max_val - min_val)

  # convert to tensor
  # cube_tensor <- torch_tensor(cube_array_seq, dtype = torch_float())

  # rm(cube_array_seq)
  # gc()

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
  # train_data <- cube_tensor[1:train_size, , , , ]
  # val_data <- cube_tensor[(train_size+1):dim(cube_tensor)[1], , , , ]

  # max_trn <- seq_len - 2
  # min_val <- seq_len - 1

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

  #   create_val_ds <- dataset(
  #
  #     initialize = function(data) {
  #       self$data <- data
  #     },
  #
  #     .getitem = function(i) {
  #       list(x = self$data[i, 1:5, ..], y = self$data[i, 6, ..])
  #     },
  #
  #     .length = function() {
  #       nrow(self$data)
  #     }
  #   )
  #
  #   val_ds <- create_val_ds(cube_tensor)
  #   val_dl <- dataloader(val_ds, batch_size = batch_size, shuffle = FALSE)

  return(train_dl)
}

