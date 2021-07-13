test_that("Tests if RF with symmetric splits works", {
  context('Tests symmetric = TRUE flag')

  x <- iris[, c(1,2,3)]
  y <- iris[, 4]

  set.seed(275)

  # Test forestry (mimic RF)
  expect_error(
    forest <- forestry(x,y,ntree = 1,maxDepth = 1,symmetric = TRUE),
    "When running Symmetric splits, all continuous variables  must have positive and negative supports"
  )

  # We need all covariates to contain negative and positive values
  x <- apply(x, 2, function(x) {return(x - mean(x))})

  # Test predict
  context("Test trinary structure")
  forest <- forestry(x,y,ntree = 1,maxDepth = 1,symmetric = TRUE)
  y_pred <- predict(forest, x)

  expect_equal(length(unique(y_pred)), 3)
  # Mean Square Error


  # Test that the predictions are symmetric
  context("Test that the predictions are symmetric")
  expect_equal(abs(y_pred[1]-y_pred[53]),
               abs(y_pred[144]-y_pred[53]), tolerance = 1e-6)


  context("High precision test")

  skip_if_not_mac()
  expect_equal(all.equal(y_pred[1:5],
                         rep(0.9256881,5),
                         tolerance = 1e-5),
               TRUE)


  context("Test enforcing monotonicity with symmetric splits")
  forest <- forestry(x,y,ntree = 1,maxDepth = 1,symmetric = TRUE, monotonicConstraints = c(1,1,1))
  y_pred <- predict(forest, newdata = data.frame(Sepal.Length = seq(-1,1,length.out = 150),
                                                 Sepal.Width = seq(-1,1,length.out = 150),
                                                 Petal.Length = seq(-2.5,1.5,length.out = 150)))


  expect_equal(all.equal(order(y_pred), 1:150), TRUE)

})
