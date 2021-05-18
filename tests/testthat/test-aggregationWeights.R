test_that("Tests that autotuning is working correctly", {

  rf <- forestry(x = iris[,-1],
                 y = iris[,1],
                 seed = 432432)

  context("Try indices out of range")
  expect_error(p <- predict(rf, newdata = iris[,-1], trees = c(1,2,3,4,4,501)),
                 "trees must contain indices which are integers between 1 and ntree")
  expect_error(p <- predict(rf, newdata = iris[,-1], trees = c(1,2,3,4,4,0)),
                 "trees must contain indices which are integers between 1 and ntree")

  context("Try settings not allowed")
  expect_error(predict(rf, newdata = iris[,-1], trees = c(1,2,3,4,4,4.2342)))
  expect_error(predict(rf, newdata = iris[,-1], trees = c(1,2,3,4,4,4.2342), exact = FALSE))
  expect_error(predict(rf, newdata = iris[,-1], trees = c(1,2,3,4,4,4.2342), aggregation = "weightMatrix"))

  context("Try several allowed settings")
  P <- predict(rf, newdata = iris[,-1])
  P_2 <- predict(rf, newdata = iris[,-1], trees = c(1:500))
  expect_equal(all.equal(P_2, P), TRUE)

  context("Test linearity of predictions")
  P_1 <- predict(rf, newdata = iris[,-1], trees = c(1))
  P_2 <- predict(rf, newdata = iris[,-1], trees = c(2))
  P_3 <- predict(rf, newdata = iris[,-1], trees = c(3))
  P_4 <- predict(rf, newdata = iris[,-1], trees = c(4))

  P_all <- predict(rf, newdata = iris[,-1], trees = c(1,2,3,4))
  P_agg <- .25*(P_1+P_2+P_3+P_4)

  expect_equal(all.equal(P_all, .25*(P_1+P_2+P_3+P_4)), TRUE)

  context("High precision test")
  skip_if_not_mac()

  expect_equal(P_1[1:5], c(5.133333333,
                           4.658333333,
                           4.658333333,
                           5.076923077,
                           5.133333333), tolerance = 1e-8)


  context("Test giving tree indices with replacement")
  P_all <- predict(rf, newdata = iris[,-1], trees = c(1,1,1,2,2))
  expect_equal(all.equal(P_all, .2*(P_1+P_1+P_2+P_2+P_1)), TRUE)
})
