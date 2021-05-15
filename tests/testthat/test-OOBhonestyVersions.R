test_that("Tests if different OOB honesty sampling schemes are working as expected", {

  set.seed(292313)

  x_train <- iris[, -1]
  y_train <- iris[, 1]

  rf <- forestry(x = x_train,
                 y = y_train,
                 ntree = 1,
                 OOBhonest = TRUE,
                 doubleBootstrap = TRUE,
                 seed = 2312,
                 nthread = 2)

  rf <- make_savable(rf)

  preds <- predict(rf, feature.new = x_train, aggregation = "oob")

  # When we run OOBhonesty, with doubleBootstrap = TRUE, when we do oob predictions
  # we should predict for all observations in the splitting set, but not any observations in
  # the averaging set
  avg_indices <- sort(unique(rf@R_forest[[1]]$averagingSampleIndex))
  nan_prediction_indices <- which(is.nan(preds))

  expect_equal(all.equal(avg_indices,
                         nan_prediction_indices), TRUE)

  # Now when we predict with the doubleOOB aggregation, we should predict only for
  # the observations which were not in either the splitting or aggregation set
  doubleOOBindices <- setdiff(setdiff(1:nrow(x_train),
                                      rf@R_forest[[1]]$splittingSampleIndex),
                                      rf@R_forest[[1]]$averagingSampleIndex)


  preds <- predict(rf, aggregation = "doubleOOB")
  prediction_indices <- which(!is.nan(preds))

  expect_equal(all.equal(prediction_indices, doubleOOBindices),
               TRUE)

})
