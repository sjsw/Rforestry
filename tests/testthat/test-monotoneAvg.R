library(testthat)
test_that("Tests that Monotone Avg parameter is working correctly", {
  set.seed(24750371)
  # Simulate some data that should be positive monotone
  x <- data.frame(V1 = runif(100, min = 0, max = 10))
  y <- .2*x[,1] + rnorm(100)

  context('Positive monotone splits with honesty')
  # Set seed for reproductivity


  monotone_avg_forest <- forestry(
    x,
    y,
    ntree = 500,
    nodesizeStrictSpl = 5,
    nodesizeStrictAvg = 1,
    maxDepth = 10,
    nthread = 2,
    monotonicConstraints = c(1),
    monotoneAvg = TRUE,
    OOBhonest = TRUE
  )
  # Test predictions are monotonic increasing in the first feature
  # Using the monotoneAvg parameter, the predictions from the averaging set should
  # respect monotonicity
  pred_means_avg_monotone <- sapply(c(1:9), function(x) {mean(predict(monotone_avg_forest,
                                                         feature.new = data.frame(V1 = rep(x, 100))))})

  expect_equal(all.equal(order(pred_means_avg_monotone), 1:9), TRUE)

  monotone_forest <- forestry(
    x,
    y,
    ntree = 500,
    nodesizeStrictSpl = 5,
    maxDepth = 10,
    nthread = 2,
    monotonicConstraints = c(1),
    monotoneAvg = FALSE,
    OOBhonest = TRUE
  )
  # Test predictions are monotonic increasing in the first feature
  pred_means <- sapply(c(1:9), function(x) {mean(predict(monotone_forest,
                                                         feature.new = data.frame(V1 = rep(x, 100))))})

  # If we don't use the parameter, monotonicity is broken
  expect_equal(all.equal(order(pred_means), c(1,5,4,3,2,7,6,9,8)), TRUE)
})
