
download_s5p_from_openeo <- function( country = NULL,
                                      date_ext = list(),
                                      n = NULL, e = NULL, s = NULL, w = NULL,
                                      output_path=".", con = connect(host = "https://openeo.cloud"), p = processes()){

      # mask for cloud cover
      threshold_ = function(data, context) {

        threshold = p$gte(data[1], 0.5)
        return(threshold)
      }

      # interpolate where no data
      interpolate = function(data,context) {
        return(p$array_interpolate_linear(data = data))
      }

      # use bounding box
      bbox = c(n, e, s, w)
      if (is.null(country) & !is.null(bbox)){
        n = bbox$n
        e = bbox$e
        s = bbox$s
        w = bbox$w

      # use country name
      }else if(!is.null(country) & is.character(country) & is.null(bbox)){
        country_sf = rnaturalearth::ne_countries(country = country, returnclass = "sf", scale = 'large')
        n = sf::st_bbox(country_sf)[4]
        e = sf::st_bbox(country_sf)[3]
        s = sf::st_bbox(country_sf)[2]
        w = sf::st_bbox(country_sf)[1]

      # use country sf
      }else if((!is.null(country) & inherits(country, "sf") & is.null(bbox))){
        n = sf::st_bbox(country)[4]
        e = sf::st_bbox(country)[3]
        s = sf::st_bbox(country)[2]
        w = sf::st_bbox(country)[1]

      }else{
        stop(cli::format_error("Either pass a country in sf of character, or the coordinates. Never both!"))
      }

      # acquire data for the extent
      datacube_no2 = p$load_collection(
        id = "SENTINEL_5P_L2",
        spatial_extent = list(west = w, south = s, east = e, north = n),
        temporal_extent=date_ext,
        bands=c("NO2")
      )

      datacube = p$load_collection(
        id = "SENTINEL_5P_L2",
        spatial_extent = list(west = w, south = s, east = e, north = n),
        temporal_extent=date_ext,
        bands=c("CLOUD_FRACTION")
      )

      # apply the threshold to the cube
      cloud_threshold <- p$apply(data = datacube, process = threshold_)

      # mask the cloud cover with the calculated mask
      datacube <- p$mask(datacube_no2, cloud_threshold)


      datacube = p$apply_dimension(process = interpolate,
                                 data = datacube, dimension = "t"
    )

    cli::cli_rule(left = "Download S5P data", right = "{lubridate::now()}");cli::cli_end()
    formats = list_file_formats()
    result = p$save_result(data = datacube, format = formats$output$GeoTiff)
    job = create_job(graph=result, title = "s5p")
    start_job(job = job)
    jobs = list_jobs()
    while (jobs[[job$id]]$status == 'running' | jobs[[job$id]]$status == 'queued' | jobs[[job$id]]$status == 'created' ){

      print(paste0('this may take a moment, your process is ', jobs[[job$id]]$status))
      Sys.sleep(60)

      jobs = list_jobs()
      if (jobs[[job$id]]$status == 'finished' | jobs[[job$id]]$status == 'error'){
        break
      }
    }

    cli::cli_progress_bar("Downloading data", type = "download")
    try(dir.create(output_path))
    download_results(job = job$id, folder = output_path)

    cli::cli_progress_done()
    cli::cli_alert_success('Download and processing was {cli::col_green("successfull")}!')
    cli::cli_rule();cli::cli_end()
}

create_dl_from_cube <- function(path_to_tiffs = "vignettes/data/switzerland", batch_size = 100, train_pct = 0.8, seq_len = 12, device = device) {

  # Get all the file paths in the directory
  file_paths <- list.files(path = path_to_tiffs, pattern = "*\\.tif$", full.names = TRUE)

  # Read the files and combine into a single array
  cube_array <- array(0, dim = c(length(file_paths), 1, nrow(raster::raster(file_paths[[1]])), ncol(raster::raster(file_paths[[1]]))))
  for (i in seq_along(file_paths)) {
    cube_array[i,,,] <- raster(file_paths[[i]])[]
  }

  # Create sequence tensor
  cube_array_seq <- array(0, dim = c(dim(cube_array)[1] - seq_len + 1, seq_len, dim(cube_array)[2], dim(cube_array)[3], dim(cube_array)[4]))
  for (i in 1:(dim(cube_array)[1] - seq_len + 1)) {
    cube_array_seq[i, , , ,] <- cube_array[i:(i + seq_len - 1), , , ]
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



## Create Dummy Data
create_dummy_data <- function(){
  beams <- vector(mode = "list", length = 6)
  beam <- torch_eye(6) %>% nnf_pad(c(6, 12, 12, 6)) # left, right, top, bottom
  beams[[1]] <- beam

  for (i in 2:6) {
    beams[[i]] <- torch_roll(beam, c(-(i-1),i-1), c(1, 2))
  }

  init_sequence <- torch_stack(beams, dim = 1)
  sequences <- vector(mode = "list", length = 100)
  sequences[[1]] <- init_sequence

  for (i in 2:100) {
    sequences[[i]] <- torchvision::transform_random_affine(init_sequence, degrees = 0, translate = c(0.5, 0.5))
  }

  input <- torch_stack(sequences, dim = 1)

  # add channels dimension
  input <- input$unsqueeze(3)
  dim(input)

  dummy_ds <- dataset(

    initialize = function(data) {
      self$data <- data
    },

    .getitem = function(i) {
      list(x = self$data[i, 1:5, ..], y = self$data[i, 6, ..])
    },

    .length = function() {
      nrow(self$data)
    }
  )

  ds <- dummy_ds(input)
  dl <- dataloader(ds, batch_size = 100)

  return(dl)
}






