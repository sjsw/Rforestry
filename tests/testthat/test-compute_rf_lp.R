test_that("Tests that compute the lp distances works correctly", {

  context('Test lp distances')

  # Set seed for reproductivity
  set.seed(292313)

  # Use Iris Data
  test_idx <- sample(nrow(iris), 11)
  x_train <- iris[-test_idx, -1]
  y_train <- iris[-test_idx, 1]
  x_test <- iris[test_idx, -1]

  # Create a random forest
  rf <- forestry(x = x_train, y = y_train, nthread = 1)

  # Compute the l1 distances in the "Species" dimension
  distances_1 <- compute_lp(object = rf,
                            newdata = x_test,
                            feature = "Species",
                            p = 1)

  # Compute the l2 distances in the "Petal.Length" dimension
  distances_2 <- compute_lp(object = rf,
                            newdata = x_test,
                            feature = "Petal.Length",
                            p = 2)

  expect_identical(length(distances_1), nrow(x_test))
  expect_identical(length(distances_2), nrow(x_test))

  #set tolerance
  #skip_if_not_mac()

  expect_equal(distances_1,
               c(0.750188131925045, 0.552840834880335, 0.675485609753403, 0.493190460443415,
                 0.456873658763604, 0.798842364622228, 0.693076971524736, 0.636592090398513,
                 0.797671440982423, 0.574115633920879, 0.688162142792068),
               tolerance = 1e-12)
  expect_equal(distances_2,
               c(2.38988174680437, 2.48118855696082, 2.74061447114056, 1.91215898745165,
                 1.73023256577338, 2.43576791790054, 2.10072196438777, 2.45140550620141,
                 3.19121772940501, 2.44963524029775, 2.31748543164867),
               tolerance = 1e-12)
})

