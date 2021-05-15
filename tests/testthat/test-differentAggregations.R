test_that("Tests that saving RF and laoding it works", {
  library(Rforestry)

  rf <- forestry(x = iris[,-1],
                 y = iris[,1],
                 OOBhonest = TRUE)

  doubleOOBpreds <- getOOBpreds(rf, doubleOOB = TRUE,
                                noWarning = TRUE)

  OOBpreds <- getOOBpreds(rf,noWarning = TRUE)

  predict_doubleOOBpreds <- predict(rf, aggregation = "oob")

  predict_OOBpreds <- predict(rf, aggregation = "doubleOOB")

  # Expect OOB preds from getOOB preds and predict to be the same
  expect_equal(all.equal(predict_OOBpreds,
                         OOBpreds), TRUE)

  # Expect double OOB preds to be the same from predict and getOOBpreds
  expect_equal(all.equal(predict_doubleOOBpreds,
                         doubleOOBpreds), TRUE)

  expect_error(
    predict_OOBpreds <- predict(rf, aggregation = "average"),
    "When using an aggregation that is not oob or doubleOOB, one must supply newdata"
  )

})
