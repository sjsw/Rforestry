test_that("Tests that the getSplitProps() function is working", {
  x <- iris[, -1]
  y <- iris[, 1]

  context('Test that the important features are found')
  # Set seed for reproductivity
  set.seed(24750371)

  # Test forestry (mimic RF)
  forest <- forestry(
    x,
    y,
    OOBhonest = TRUE
  )

  p <- getSplitProps(forest)

  expect_gt(unname(p["Sepal.Width"]), unname(p["Species"]))

  skip_if_not_mac()
  context("Test exact values of split proportions")
  expect_equal(unname(p["Sepal.Width"]),  0.3274834)

})
