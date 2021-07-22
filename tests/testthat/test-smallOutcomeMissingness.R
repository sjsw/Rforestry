library(testthat)
test_that("Tests missing value splitting when Outcome values are close to zero", {
  # Set seed for reproductivity
  set.seed(24750371)


  context("Test Missing data + Small Y for underflow")
  x <- data.frame(V1 = runif(5000, min = -1e-8, max = 1e-8),
                  V2 = runif(5000, min = -1e-8, max = 1e-8),
                  V3 = runif(5000, min = -1e-8, max = 1e-8))

  y <- 5.7*x$V3

  # print(summary(y))
  # print(summary(y^2))

  # Take some missing examples
  sample_idx_1 <- sample(1:5000, size = 2000, replace = FALSE)
  x[sample_idx_1,1] <- NA

  sample_idx_2 <- sample(1:5000, size = 2000, replace = FALSE)
  x[sample_idx_2,2] <- NA


  forest <- forestry(
    x,
    y,
    nthread = 2,
    seed = 2
  )

  # In this example, we speculate that if the forest starts sending missing
  # observations in random directions, that will mess up the R^2

  p <- predict(forest, x, seed = 12)
  R_squared <- 1- (mean((p - y)^2))/(var(y))

  # print(R_squared)

  # WHEN USING ABS(), R^2 is pretty high, should be able to catch most of the
  # signal here despite a lot of missingness (should be > .65)
  #  0.9624975 for square()
  #  0.9624975 for std::abs()
  #  0.9624975 for std::fabs()
  #  0.9624975 for abs()
  #  0.9624975 for squaring by hand

  expect_gt(R_squared, 0.65)
})
