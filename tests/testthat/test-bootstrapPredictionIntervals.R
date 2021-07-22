test_that("Tests that computing confidence intervals is going well", {
  x <- iris[1:125,-1]
  y <- iris[1:125,1]

  x_test <- iris[126:150,-1]
  y_test <- iris[126:150,1]

  rf <- forestry(x = x,
                 y = y,
                 OOBhonest = TRUE,
                 seed = 3242)

  context("test the bootstrapped prediction intervals")
  preds <- getCI(rf, newdata = x_test, level = .95, method = "OOB-bootstrap")

  coverage <- length(which(y_test < preds$CI.upper & y_test > preds$CI.lower)) / length(y_test)

  skip_if_not_mac()
  expect_gt(coverage, 0)

})
