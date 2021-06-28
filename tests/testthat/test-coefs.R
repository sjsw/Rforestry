test_that("Tests if ridgeRF works when using coefficient aggregation", {
  context('Tests aggregation = \"coefs\" ')

  x <- iris[, c(1,2,3)]
  y <- iris[, 4]

  # Check if coefficient aggregation is working well
  set.seed(231428176)
  forest <- forestry(
    x,
    y,
    ntree = 200,
    linear = TRUE,
    overfitPenalty = 1
  )
  y_pred <- predict(forest, x, aggregation = "coefs")

  x_mat <- as.matrix(cbind(x, Int = 1))
  preds_using_coefs <- rowSums(x_mat*y_pred$coef)
  expect_equal(all.equal(y_pred$predictions, preds_using_coefs), TRUE)
})
