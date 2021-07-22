test_that("Tests that new version of missing splits is working correctly", {
library(Rforestry)


  context("test putting all NA's to right")
  # Test example with simple step function
  x <- rnorm(100)
  y <- ifelse(x > 0, 1,0) + rnorm(100, mean = 0, sd = .1)
  x <- data.frame(x)

  # plot(x$x, y)

  # Only make right observations missing now
  missing_idx <- sample(which(x$x > 0), size = 10, replace = FALSE)
  x$x[missing_idx] <- NA

  rf <- forestry(x = x,
                 y = y,
                 maxDepth = 1)

  # Now RF should push missing obs to the right
  preds <- predict(rf, newdata = data.frame(x = rep(NA, 10)))

  expect_equal(any(preds < .5), FALSE)

  # Test example with simple step function
  x <- rnorm(100)
  y <- ifelse(x > 0, 1,0) + rnorm(100, mean = 0, sd = .1)
  x <- data.frame(x)

  # plot(x$x, y)


  context("test putting all NA's to left")
  # Only make left observations missing now
  missing_idx <- sample(which(x$x < 0), size = 10, replace = FALSE)
  x$x[missing_idx] <- NA

  rf <- forestry(x = x,
                 y = y,
                 maxDepth = 1)

  # Now RF should push missing obs to the right
  preds <- predict(rf, newdata = data.frame(x = rep(NA, 10)))

  expect_equal(any(preds > .5), FALSE)



  context("test putting more NA's to right than left")
  # Now say we have a mix of missingness, but we still have more missing right than left


  # Test example with simple step function
  x <- rnorm(100)
  y <- ifelse(x > 0, 1,0) + rnorm(100, mean = 0, sd = .1)
  x <- data.frame(x)

  # plot(x$x, y)

  # Only make left observations missing now
  missing_idx_1 <- sample(which(x$x > 0), size = 7, replace = FALSE)
  missing_idx_2 <- sample(which(x$x < 0), size = 3, replace = FALSE)
  missing_idx <- c(missing_idx_1, missing_idx_2)
  x$x[missing_idx] <- NA

  rf <- forestry(x = x,
                 y = y,
                 maxDepth = 1)

  # Now RF should push missing obs to the right
  preds <- predict(rf, newdata = data.frame(x = rep(NA, 10)))

  expect_equal(any(preds < .5), FALSE)


  context("Expect all splits to send all NA's to the same side")

  # Now run RF with a pure noise outcome, we should still send all NA's to either left or right
  rf <- forestry(x = x,
                 y = rnorm(100),
                 ntree = 15,
                 maxDepth = 1)

  rf <- make_savable(rf)

  noSplitNA <- TRUE
  for (i in 1:15) {
    # For all trees we should only have NA obs going left or right
    if ((rf@R_forest[[i]]$naRightCounts[1] > 0) &&
        (rf@R_forest[[i]]$naLeftCounts[1] > 0)) {
      noSplitNA <- FALSE
    }
  }

  expect_equal(noSplitNA, TRUE)

})
