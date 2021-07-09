test_that("Tests if RF with symmetric splits works", {
  context('Tests symmetric = TRUE flag')

  x <- iris[, c(1,2,3)]
  y <- iris[, 4]

  set.seed(275)

  # Test forestry (mimic RF)
  forest <- forestry(
    x,
    y,
    ntree = 1,
    maxDepth = 1,
    symmetric = TRUE
  )

  # Test predict
  y_pred <- predict(forest, x)

  # Mean Square Error
  mean((y_pred - y) ^ 2)
})
