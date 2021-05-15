test_that("Tests if OOB predictions are working with supplied feature", {
  x <- iris[, -1]
  y <- iris[, 1]
  context('OOB Predictions using feature.new')
  # Set seed for reproductivity
  set.seed(24750371)

  rf <- forestry(x=x,
                 y=y)

  change_species_x <- iris[,-1]

  # Shift a feature in the original train data
  change_species_x$Petal.Width[1:50] = change_species_x$Petal.Width[1:50] + 2


  preds_modified <- getOOBpreds(rf, newdata = change_species_x,
                                noWarning = TRUE)
  preds_original <- getOOBpreds(rf,
                                noWarning = TRUE)

  skip_if_not_mac()

  expect_equal(all.equal(preds_modified[51:150],
                         preds_original[51:150]), TRUE)

  expect_gt(mean(preds_modified[1:50]), mean(preds_original[1:50]))

  # Now try testing an error
  k <- iris[1:10,-1]
  expect_warning(
    preds <- getOOBpreds(rf, newdata = k)
  )
  expect_equal(preds, NA)

})



