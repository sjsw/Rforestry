test_that("Tests that computing conformal local intervals is working", {
  x <- iris[1:125,-1]
  y <- iris[1:125,1]

  x_test <- iris[126:150,-1]
  y_test <- iris[126:150,1]

  rf <- forestry(x = x,
                 y = y,
                 OOBhonest = TRUE)

  context("test local conformal intervals")
  preds <- getCI(rf, newdata = x_test, level = .95, method = "local-conformal")

  coverage <- length(which(y_test < preds$CI.upper & y_test > preds$CI.lower)) / length(y_test)

  expect_gt(coverage, .8)

})
