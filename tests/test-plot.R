library(testthat)

set.seed(1)
n <- c(100)
a <- rnorm(n)
b <- rnorm(n)
c <- rnorm(n)
y <- c(4*a + 5.5*b - .78*c, rep(0, 10))
a <- c(a, rep(0, 10))
b <- c(b, rep(0, 10))
c <- c(c, rep(0, 10))

x <- data.frame(a,b,c)

test_that("Tests that ridgeRF plotting does not generate errors with a single linFeat", {
  forest <- forestry(
    x,
    y,
    ntree = 10,
    replace = TRUE,
    maxDepth = 100,
    nodesizeStrictSpl = 10,
    nodesizeStrictAvg = 10,
    nthread = 2,
    linear = TRUE,
    linFeats = 1
  )
  expect_silent(plot(forest))
})


test_that("Tests that ridgeRF plotting does not generate errors when leaves are grown to purity", {
  forest <- forestry(
    x,
    y,
    ntree = 10,
    replace = TRUE,
    maxDepth = 100,
    nodesizeStrictSpl = 1,
    nodesizeStrictAvg = 1,
    nthread = 2,
    linear = TRUE,
    linFeats = 1
  )
  expect_silent(plot(forest))
})

