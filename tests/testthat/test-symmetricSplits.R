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
  context("High precision test")

  skip_if_not_mac()
  expect_equal(all.equal(y_pred[1:5],
                         c(0.2395833, 0.2395833, 0.2395833, 0.2395833, 0.2395833),
                         tolerance = 1e-5),
               TRUE)
})
