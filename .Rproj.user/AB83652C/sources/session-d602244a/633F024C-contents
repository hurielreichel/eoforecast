# NO2 Forecasting using Sentinel 5P and Weather Forecast data: a Deep Learning Approach
library(eoforecast)

## S5P Data Download & Wrangling
# open openeo connection
con = connect(host = "https://openeo.cloud")

# login
login()

# download s5p
download_s5p_from_openeo(country = "switzerland",
                         date_ext = c("2019-01-01", "2023-01-01"),
                         output_path="vignettes/data/switzerland/")

# Read as Array
cube = create_dl_from_cube()
cube %>% readr::write_rds("vignettes/data/switzerland/cube_ch.rds")


