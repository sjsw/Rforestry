test_that("Tests if OOB honesty with the optional doubleBootstrap flag is working properly", {

  set.seed(292313)


  x_train <- iris[, -1]
  y_train <- iris[, 1]

  rf <- forestry(x = x_train,
                 y = y_train,
                 ntree = 1,
                 OOBhonest = TRUE,
                 nodesizeStrictAvg = 1,
                 nthread = 2)

  rf <- make_savable(rf)

  # Check there is no overlap in splitting/ averaging observations
  expect_equal(length( intersect( (rf@R_forest[[1]]$splittingSampleIndex),
                                  (rf@R_forest[[1]]$averagingSampleIndex))),
               0)

  # When the flag is on, we should have some duplicates in the averaging set
  expect_equal(any(duplicated(rf@R_forest[[1]]$averagingSampleIndex)),
               TRUE)


  context("Try the doubleBootstrap flag off")

  rf <- forestry(x = x_train,
                 y = y_train,
                 ntree = 1,
                 OOBhonest = TRUE,
                 doubleBootstrap = FALSE,
                 nodesizeStrictAvg = 1,
                 nthread = 2)

  rf <- make_savable(rf)

  # Still should be disjoint
  expect_equal(length( intersect( (rf@R_forest[[1]]$splittingSampleIndex),
                                  (rf@R_forest[[1]]$averagingSampleIndex))),
               0)



  # When the flag is off, we should have no duplicates as the OOB set
  # is not resampled again
  expect_equal(any(duplicated(rf@R_forest[[1]]$averagingSampleIndex)),
               FALSE)


})
