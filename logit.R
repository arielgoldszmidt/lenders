setwd('Documents/School/Year 3/Spring/ECON 21150/lenders/')

train_features <- read.csv('train_features.csv')
train_labels <- read.csv('train_labels.csv')

train_features$X <- NULL
models <- c()

for (g in c('A','B','C','D','E','F','G')) {
    data <- train_features[train_labels[, paste('grade_', g, sep = '')] == 1,]
    data$bad <- train_labels[train_labels[, paste('grade_', g, sep = '')] == 1,]$bad
    
    models <- c(models, glm(bad ~ ., data = data, family = 'binomial'))
}

do.call(stargazer, list(models))

