test_that("Tests that adaptive Forestry works", {
  set.seed(292313)

  test_idx <- sample(nrow(iris), 11)
  x_train <- iris[-test_idx, -1]
  y_train <- iris[-test_idx, 1]
  x_test <- iris[test_idx, -1]

  context("Train adaptiveForestry")
  rf <- adaptiveForestry(x = x_train,
                          y = y_train,
                          ntree.first = 25,
                          ntree.second = 500,
                          nthread = 2)
  p <- predict(rf@second.forest, x_test)

  expect_equal(length(p), 11)

  context("High precision test for prediction of adaptiveForestry")
  skip_if_not_mac()

  expect_equal(all.equal(p, c(6.631921, 6.591024, 4.976911, 4.824026, 6.061833,
                              5.018555, 5.097299, 6.138423, 6.462167,
                              5.709681, 6.495333), tolerance = 1e-6), TRUE)

})
