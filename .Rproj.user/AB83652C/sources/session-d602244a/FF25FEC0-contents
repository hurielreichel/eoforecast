## convLSTM
library(eoforecast)
# dummy input tensor
x <- torch_rand(c(2, 4, 3, 16, 16)) # batch_size, seq_len, channels, height, width

## Sanity Check
dl <- create_dummy_data()
preds <- train_convlstm(dl = dl, 10, plot_path = "vignettes/learning_curve-sanity.png")

## Wrangle Cube to tensor
cube <- readRDS("vignettes/data/switzerland/cube_ch.rds")
train_dl <- cube[1]
test_dl <- cube[2]

preds <- train_convlstm(train_dl, 3)



