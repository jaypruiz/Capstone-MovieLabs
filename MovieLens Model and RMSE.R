################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

################################
# Clean edx partition for model
################################

edx_part <- edx%>%
  group_by(userId)%>% 
  filter(n() >=40) %>% ungroup()
#Filters edx such that it only considers users with more than 40 reviews

#####################################
# Matrix Factorization via Recosystem
#####################################

train_data <- data_memory(user_index = edx_part$userId, 
                          item_index= edx_part$movieId,
                          rating=edx_part$rating, index1= TRUE)
#recosystem requires arguments to be in data_memory or data_file format.
#We set the indeces manually, as edx_part is not in the default format.
#recosystem develops the algorithm through matrix factorization on the
#edx set we partitioned
test_data <- data_memory(user_index = validation$userId, 
                         item_index= validation$movieId,
                         rating=validation$rating, index1= TRUE)
#test_data is WHAT WE ARE RUNNING THE ALGORITHM ON. This is not
#overtraining or using the validation set to train. If you mistakenly
#make a test and train partition of edx and then put the test in
#prediction_model$predict, you will use the algorithm on the new
#partition instead of on the validation set, so your RMSE will fail.
prediction_model <- Reco()
#Reco() creates a special object used by recosystem. 
set.seed(3080) 
#recosystem's methods involve randomization, so a seed is needed
prediction_model$train(train_data, 
                       opts=c(dim=35, costp_l2=.1, costq_l2=.1,
                              lrate=.1, niter=300, nthread=8, verbose=F))
#dim is the number of factors we are testing, p and q regularization 
#settings are costp_l2 and costq_l2 (we want to optimize these as much 
#as possible), niter is the number of iterations and lrate is the 
#learning rate. If you have too high of a learning rate, your algorithm's
#model will extremely normalize. nthread is a performance setting. 
#These options are not required, but through trial and error, 
#these got good results,sensibly by the number of iterations being used.
validation_prediction <- prediction_model$predict(test_data, out_memory())
#out_memory() is the function that enables that transposes the data_memory
#back into a dataframe, which is what we need in order to run RMSE.

####################
# Calculate the RMSE
####################

RMSE(validation$rating, validation_prediction)
#RMSE that comes with caret requires two dataframes for arguments.