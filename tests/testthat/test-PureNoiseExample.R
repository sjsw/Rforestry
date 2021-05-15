test_that("Tests if OOB honesty is able to have low correlation to the training data in a pure noise response", {
  dta <- data.frame(rep = 1:3)
  dta$OOB <- NA
  dta$OOBdoubleBoot <- NA


  for (rep_i in 1:3) {
    set.seed(rep_i + 2342)
    n=1e3
    X0 <- matrix(nrow=n,ncol=1)
    X0[,1] <- rnorm(n)
    colnames(X0) <- "X0"
    Y0 <- rnorm(n)

    f0 <- forestry(y=Y0,
                   x=X0,
                   OOBhonest = TRUE,
                   doubleBootstrap = FALSE
    )

    p0oob <- predict(f0, aggregation = "oob")


    dta$OOB[which(dta$rep == rep_i)] <- cor(p0oob,Y0)

    f0 <- forestry(y=Y0,
                   x=X0,
                   OOBhonest = TRUE,
                   doubleBootstrap = TRUE
    )

    p0oob <- predict(f0, aggregation = "oob")

    dta$OOBdoubleBoot[which(dta$rep == rep_i)] <- cor(p0oob,Y0)
  }

  mean <- unname(apply(dta[,-1],2,mean))

  # Expect both correlations to be quite small when we use OOB predictions
  expect_lt(abs(mean[1]), .05)

  expect_lt(abs(mean[2]), .05)


})
