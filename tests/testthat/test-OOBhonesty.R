test_that("Tests if OOB Honesty is working correctly", {
  x <- iris[, -1]
  y <- iris[, 1]
  context('OOB Honesty')
  # Set seed for reproductivity
  set.seed(24750371)

  # Test forestry (mimic RF)
  forest <- forestry(
    x,
    y,
    ntree = 500,
    replace = TRUE,
    sampsize = nrow(x),
    mtry = 3,
    nodesizeStrictSpl = 5,
    nthread = 2,
    splitrule = "variance",
    splitratio = 1,
    OOBhonest = TRUE,
    seed = 8921,
    nodesizeStrictAvg = 0
  )

  # Test OOB
  skip_if_not_mac()
  expect_lt(mean((getOOB(forest) - 12.73149)^2), .1)

  # Test what happens when we specify splitratio as well as OOBhonest
  expect_warning(forest <- forestry(
    x,
    y,
    ntree = 500,
    replace = TRUE,
    sampsize = nrow(x),
    mtry = 3,
    nodesizeStrictSpl = 5,
    nthread = 2,
    splitrule = "variance",
    nodesizeStrictAvg = 1,
    splitratio = .4,
    OOBhonest = TRUE
  ),"OOBhonest is set to true, so we will run OOBhonesty rather
            than standard honesty"
  )

  skip_if_not_mac()
  expect_equal(getOOBpreds(forest)[1:3], c(5.048432, 4.698436, 4.698634), tolerance = 1e-3)


  context('Test OOB Honesty vs honest OOB set')
  # Now test that forestry is getting the right OOB indices when using OOBhonest vs normal honesty
  forest <- forestry(
    x,
    y,
    ntree = 1,
    replace = TRUE,
    sampsize = nrow(x),
    mtry = 3,
    nodesizeStrictSpl = 5,
    nthread = 2,
    splitrule = "variance",
    splitratio = 1,
    OOBhonest = TRUE,
    seed = 8921,
    nodesizeStrictAvg = 0,
    saveable = TRUE
  )
  forest <- make_savable(forest)

  skip_if_not_mac()
  # OOB preds should be for only observations in the splitting set for the tree
  expect_equal(sort(which(!is.nan(getOOBpreds(forest)))),
               sort(unique(forest@R_forest[[1]]$splittingSampleIndex)))

  forest <- forestry(
    x,
    y,
    ntree = 1,
    replace = TRUE,
    sampsize = nrow(x),
    mtry = 3,
    nodesizeStrictSpl = 5,
    nthread = 2,
    splitrule = "variance",
    splitratio = .5,
    OOBhonest = FALSE,
    seed = 8921,
    nodesizeStrictAvg = 0,
    saveable = TRUE
  )
  forest <- make_savable(forest)

  # OOB preds should be for only observations in the splitting set for the tree
  oob_index <- setdiff(1:150,
                       union(forest@R_forest[[1]]$splittingSampleIndex,
                             forest@R_forest[[1]]$averagingSampleIndex))

  skip_if_not_mac()
  expect_equal(sort(which(!is.nan(getOOBpreds(forest)))),
               sort(oob_index))

  context('Test saving and loading with OOB Honesty')
  # Test that saving and loading
  # -- Actual saving and loading -----------------------------------------------
  forest <- forestry(
    x,
    y,
    sample.fraction = 1,
    splitratio = 1,
    ntree = 3,
    nthread = 2,
    saveable = TRUE,
    replace = TRUE
  )
  forest <- make_savable(forest)

  wd <- tempdir()

  # y_pred_before <- getOOBpreds(forest)

  # saveForestry(forest, filename = file.path(wd, "forest.Rda"))
  # rm(forest)
  # forest_after <- loadForestry(file.path(wd, "forest.Rda"))
  #
  # y_pred_after <- getOOBpreds(forest_after)
  # testthat::expect_equal(y_pred_before, y_pred_after, tolerance = 1e-6)

})
