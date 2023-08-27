library('caret')
library('tidyverse')
library('data.table')
library('ggplot2')

# Preparing the Data
movies <- read.csv("/Users/kaitlynyeh/Downloads/Side Projects/movies.csv")
ratings <- read.csv("/Users/kaitlynyeh/Downloads/Side Projects/ratings.csv")

moviesframe <- as.data.frame(movies) %>% 
  mutate(movieId = as.numeric(levels(movieId)) [movieId],
         title = as.character(title), genres = as.character(genres))
movielens <- left_join(ratings, movies, by = 'movieId')

set.seed(8)
indextest <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-indextest,]
samp <- movielens[indextest,]

confirmation <- samp %>% semi_join(edx, by = 'movieId') %>% 
  semi_join(edx, by = 'userId')
removed <- anti_join(samp, confirmation)
edx <- rbind(edx, removed)

set.seed(8)
indextest <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train <- edx[-indextest,]
samp <- edx[indextest,]

test <- samp %>% semi_join(train, by = 'movieId') %>% 
  semi_join(train, by = 'userId')
removed <- anti_join(samp, test)
train <- rbind(train, removed)

# Most movies are classified into multiple genres
# The number of genres in each movie
tibble(count = str_count(edx$genres, fixed('|')), genres = edx$genres) %>% 
  group_by(count, genres) %>%
  summarise(n = n()) %>%
  arrange(-count) %>%
  head()

# The number of movies under each rating
edx %>% group_by(rating) %>% summarize(n = n())
plot(edx %>% group_by(rating) %>% summarize(n = n()), 
     main = "Number of Movies per Rating", ylab = "Count of Movies", 
     type = "o", cex = 0.5)

# the number of ratings under each movie
edx %>% group_by(movieId) %>% summarize(n = n()) %>%
  ggplot(aes(n)) + ggtitle('Number of Ratings under a Movie') + 
  xlab('Number of Ratings') + ylab('Number of Movies') +
  geom_histogram(color = 'black') + scale_x_log10()

# the number of movies users rated
edx %>% group_by(userId) %>% summarize(n = n()) %>% arrange(n)
edx %>% group_by(userId) %>% summarize(n = n()) %>%
  ggplot(aes(n)) + ggtitle('Number of Movies Users Rated') + 
  xlab('Number of Ratings') + ylab('Number of Users') +
  geom_histogram(color = 'black') + scale_x_log10()

train <- train %>% select(userId, movieId, rating, title)
test <- test %>% select(userId, movieId, rating, title)

# Mean Absolute Error
MAE <- function(trueratings, predictedratings) {
  mean(abs(trueratings - predictedratings))
}

# Mean Squared Error
MSE <- function(trueratings, predictedratings) {
  mean((trueratings - predictedratings) ^ 2)
}

# Root Mean Squared Squared Error
RMSE <- function(trueratings, predictedratings) {
  sqrt(mean((trueratings - predictedratings) ^ 2))
}

# Predicting the ratings
set.seed(1)
p <- function(x, y) mean(x == y)
rating <- seq(0.5, 5, 0.5)

Monte <- replicate(10^3, {
  s <- sample(train$rating, 100, replace = TRUE)
  sapply(rating, p, y = s)
})
p2 <- sapply(1:nrow(Monte), function(x) mean(Monte[x,]))

yhat <- sample(rating, size = nrow(test), replace = TRUE, prob = p2)
result <- tibble(Method = 'Project Goal', RMSE = 0.861, MSE = NA, MAE = NA)
result <- bind_rows(result, tibble(Method = 'Random Prediction', 
                                   RMSE = RMSE(test$rating, yhat), 
                                   MSE = MSE(test$rating, yhat), 
                                   MAE = MAE(test$rating, yhat)))
result

# Initial Prediction
mu <- mean(train$rating)
result <- bind_rows(result, tibble(Method = 'Mean Prediction', 
                                   RMSE = RMSE(test$rating, mu), 
                                   MSE = MSE(test$rating, mu), 
                                   MAE = MAE(test$rating, mu)))
result

# Include the movie bias
bias <- train %>% group_by(movieId) %>% summarize(b_i = mean(rating - mu))
bias

bias %>% ggplot(aes(x = b_i)) + ggtitle('Movie Bias') + xlab('Movie Bias') + 
  ylab('Count') + geom_histogram(col = I('black'), bins = 12) + 
  scale_y_continuous(labels = comma)

# Prediction of rating with mean and movie bias
yhatbias <- mu + test %>% left_join(bias, by = 'movieId') %>% .$b_i
result <- bind_rows(result, tibble(Method = 'Mean Movie Bias Prediction', 
                                   RMSE = RMSE(test$rating, yhatbias), 
                                   MSE = MSE(test$rating, yhatbias), 
                                   MAE = MAE(test$rating, yhatbias)))
result

# Include the user bias
biasu <- train %>% left_join(bias, by = 'movieId') %>% group_by(userId) %>% 
  summarize(b_u = mean(rating - mu - b_i))
biasu %>% ggplot(aes(x = b_u)) + ggtitle('User Bias') + xlab('User Bias') + 
  ylab('Count') + geom_histogram(col = I('black'), bins = 12) + 
  scale_y_continuous(labels = comma)

# Prediction of rating with mean, movie and user bias
yhatbiasu <- test %>% left_join(bias, by = 'movieId') %>% 
  left_join(biasu, by = 'userId') %>% mutate(pred = mu + b_i + b_u) %>% .$pred
result <- bind_rows(result, tibble(Method = 'Mean Movie and User Bias Prediction', 
                                   RMSE = RMSE(test$rating, yhatbiasu), 
                                   MSE = MSE(test$rating, yhatbiasu), 
                                   MAE = MAE(test$rating, yhatbiasu)))
result

# Residual Differences (Top 10)
train %>% left_join(bias, by = 'movieId') %>% 
  mutate(residual = rating - (mu + b_i)) %>% 
  arrange(desc(abs(residual))) %>% slice(1:10)

# Ratings for the best Movies (Top 10)
title <- train %>% select(movieId, title) %>% distinct()
train %>% left_join(bias, by = 'movieId') %>% 
  arrange(desc(b_i)) %>% group_by(title) %>%
  summarize(n = n()) %>% slice(1:10)

# Now we find the best value to minimize RMSE to regularize user and movie bias
regularize <- function(lambda, trainset, testset) {
  mu <- mean(train$rating)
  b_i <- trainset %>% group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu)/(n() + lambda))
  b_u <- trainset %>% left_join(b_i, by = 'movieId') %>%
    filter(!is.na(b_i)) %>% group_by(userId) %>% 
    summarize(b_u = sum(rating - b_i - mu)/(n() + lambda))
  predratings <- testset %>% left_join(b_i, by = 'movieId') %>%
    left_join(b_u, by = 'userId') %>% 
    filter(!is.na(b_i), !is.na(b_u)) %>% 
    mutate (pred = mu + b_i + b_u) %>% pull (pred)
  return(RMSE(predratings, testset$rating))
}

lambdas <- seq(0, 10, 0.25)
lambdarmse <- sapply(lambdas, regularize, trainset = train, testset = test)
tibble(Lambda = lambda, RMSE = lambdarmse) %>% ggplot(aes(x = Lambda, y = RMSE)) +
  geom_point() + ggtitle('Regularization')

# Apply best penalty factor lambda
lambda <- lambdas[which.min(lambdarmse)]
mu <- mean(train$rating)
b_i <- train %>% group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n() + lambda))
b_u <- train %>% left_join(b_i, by = 'movieId') %>%
  group_by(userId) %>% 
  summarize(b_u = sum(rating - b_i - mu)/(n() + lambda))
predreg <- test %>% left_join(b_i, by = 'movieId') %>%
  left_join(b_u, by = 'userId') %>% 
  filter(!is.na(b_i), !is.na(b_u)) %>% 
  mutate (pred = mu + b_i + b_u) %>% pull (pred)
result <- bind_rows(result, tibble(Method = 'Regularized Movie and User Bias', 
                                   RMSE = RMSE(test$rating, predreg), 
                                   MSE = MSE(test$rating, predreg), 
                                   MAE = MAE(test$rating, predreg)))
result

# Validation of Training and Testing
muedx <- mean(edx$rating)
biedx <- edx %>% group_by(movieId) %>% 
  summarize(b_i = sum(rating - muedx)/(n() + lambda))
buedx <- edx %>% left_join(biedx, by = 'movieId') %>% group_by(userId) %>% 
  summarize(b_u = sum(rating - b_i - muedx)/(n() + lambda))
yhatedx <- confirmation %>% left_join(biedx, by = 'movieId') %>%
  left_join(buedx, by = 'userId') %>% mutate(pred = muedx + b_i + b_u) %>% 
  pull(pred)
result <- bind_rows(result, tibble(Method = 'Last Regularization', 
                                   RMSE = RMSE(confirmation$rating, yhatedx), 
                                   MSE = MSE(confirmation$rating, yhatedx), 
                                   MAE = MAE(confirmation$rating, yhatedx)))
result

# Top 5 Movies
confirmation %>% left_join(biedx, by = 'movieId') %>%
  left_join(buedx, by = 'userId') %>% mutate(pred = muedx + b_i + b_u) %>% 
  arrange(-pred) %>% group_by(title) %>% select(title) %>% head(5)

# Bottom 5 Movies
confirmation %>% left_join(biedx, by = 'movieId') %>%
  left_join(buedx, by = 'userId') %>% mutate(pred = muedx + b_i + b_u) %>% 
  arrange(pred) %>% group_by(title) %>% select(title) %>% head(5)




