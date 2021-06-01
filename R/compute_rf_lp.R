#' @include forestry.R

# ---Computing lp distances-----------------------------------------------------
#' comptute_lp
#' @name compute_lp-forestry
#' @title compute lp distances
#' @rdname compute_lp-forestry
#' @description Return the L_p norm distances of selected test observations
#'   relative to the training observations which the forest was trained on.
#' @param object A `forestry` object.
#' @param newdata A data frame of test predictors.
#' @param feature A string denoting the dimension for computing lp distances.
#' @param p A positive real number determining the norm p-norm used.
#' @return A vector of the lp distances.
#' @examples
#'
#' # Set seed for reproductivity
#' set.seed(292313)
#'
#' # Use Iris Data
#' test_idx <- sample(nrow(iris), 11)
#' x_train <- iris[-test_idx, -1]
#' y_train <- iris[-test_idx, 1]
#' x_test <- iris[test_idx, -1]
#'
#' rf <- forestry(x = x_train, y = y_train,nthread = 2)
#' predict(rf, x_test)
#'
#' # Compute the l2 distances in the "Petal.Length" dimension
#' distances_2 <- compute_lp(object = rf,
#'                           newdata = x_test,
#'                           feature = "Petal.Length",
#'                           p = 2)
#' @export
compute_lp <- function(object, newdata, feature, p){

  # Checks and parsing:
  if (class(object) != "forestry") {
    stop("The object submitted is not a forestry random forest")
  }
  newdata <- as.data.frame(newdata)
  train_set <- slot(object, "processed_dta")$processed_x

  if (!(feature %in% colnames(train_set))) {
    stop("The submitted feature is not in the set of possible features")
  }

  # Compute distances
  y_weights <- predict(object = object,
                       newdata = newdata,
                       aggregation = "weightMatrix")$weightMatrix

  if (is.factor(newdata[1, feature])) {
    # Get categorical feature mapping
    mapping <- slot(object, "categoricalFeatureMapping")

    # Change factor values to corresponding integer levels
    #TODO(Rowan): This is specific for the iris data sets.
    # * Run it on data set with several factor valued covariates
    # * Implement a test with a data set with two factor-valued column
    # * Try to use:
    # processed_x <- preprocess_testing(newdata,
    #                                   object@categoricalFeatureCols,
    #                                   object@categoricalFeatureMapping)
    factor_vals <- mapping[[1]][2][[1]]
    map <- function(x) {
      return(which(factor_vals == x)[1])
    }
    newdata[ ,feature] <- unlist(lapply(newdata[,feature], map))

    diff_mat <- matrix(newdata[,feature],
                       nrow = nrow(newdata),
                       ncol = nrow(train_set),
                       byrow = TRUE) !=
                matrix(train_set[,feature],
                       nrow = nrow(newdata),
                       ncol = nrow(train_set),
                       byrow = FALSE)
    diff_mat[diff_mat] <- 1
  } else {
    diff_mat <- matrix(newdata[,feature],
                       nrow = nrow(newdata),
                       ncol = nrow(train_set),
                       byrow = TRUE) -
                matrix(train_set[,feature],
                       nrow = nrow(newdata),
                       ncol = nrow(train_set),
                       byrow = FALSE)
  }

  # Raise absoulte differences to the pth power
  diff_mat <- abs(diff_mat) ^ p

  # Compute final Lp distances
  distances <- apply(y_weights * diff_mat, 1, sum) ^ (1 / p)

  # Ensure that the Lp distances for a factor are between 0 and 1
  if (is.factor(newdata[1, feature])) {
    distances[distances < 0] <- 0
    distances[distances > 1] <- 1
  }

  return(distances)
}



