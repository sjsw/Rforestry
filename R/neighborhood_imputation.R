#' @title Feature imputation using random forests neighborhoods
#' @name impute_features
#' @description This function uses the neighborhoods implied by a random forest
#'   to impute missing features. The neighbors of a data point are all the
#'   training points assigned to the same leaf in at least one tree in the
#'   forest. The weight of each neighbor is the fraction of trees in the forest
#'   for which it was assigned to the same leaf. We impute a missing feature
#'   for a point by computing the weighted average feature value, using
#'   neighborhood weights, using all of the point's neighbors.
#' @param object an object of class `forestry`
#' @param newdata the feature data.frame we will impute missing features for.
#' @param seed a random seed passed to the predict method of forestry
#' @param use_mean_imputation_fallback if TRUE, mean imputation (for numeric
#'   variables) and mode imputation (for factor variables) is used for missing
#'   features for which all neighbors also had the corresponding feature
#'   missing; if FALSE these missing features remain NAs in the data frame
#'   returned by `impute_features`.

#' @return A data.frame that is newdata with imputed missing values.

#' @examples
#' iris_with_missing <- iris
#' idx_miss_factor <- sample(nrow(iris), 25, replace = TRUE)
#' iris_with_missing[idx_miss_factor, 5] <- NA
#' idx_miss_numeric <- sample(nrow(iris), 25, replace = TRUE)
#' iris_with_missing[idx_miss_numeric, 3] <- NA
#'
#' x <- iris_with_missing[,-1]
#' y <- iris_with_missing[, 1]
#'
#' forest <- forestry(x, y, ntree = 500, seed = 2,nthread = 2)
#' imputed_x <- impute_features(forest, x, seed = 2)
#' @export
impute_features <- function(object, newdata,
                            seed = round(runif(1)*10000),
                            use_mean_imputation_fallback = FALSE) {
  # Sanity checking
  features.train <- object@processed_dta$processed_x
  if(ncol(features.train) != ncol(newdata)) {
    stop("Training data and imputation data have a different number of columns")
  }
  if(!all(colnames(newdata) == colnames(features.train))) {
    stop("newdata and training features have discordant names.")
  }
  if(!any(is.na(newdata))) {
    message("No values in newdata were missing. Returning the data without modification")
    return(newdata)
  }
  imputed_data <- rcpp_cppImputeInterface(
    object@forest,
    newdata,
    seed
  )
  for(catCol in object@categoricalFeatureMapping) {
    imputed_data[[catCol$categoricalFeatureCol]] <- factor(
      imputed_data[[catCol$categoricalFeatureCol]],
      levels = catCol$numericFeatureValues,
      labels = catCol$uniqueFeatureValues)
  }
  imputed_data <- as.data.frame(imputed_data)
  names(imputed_data) <- names(newdata)

  if(use_mean_imputation_fallback & any(is.na(imputed_data))) {
    message("Some missing observations had empty neighborhoods or consisted only of other missing observations.",
    "Using mean imputation for these.")
    imputed_data <- lapply(imputed_data, function(x) {
        replace(x, is.na(x),
                if(is.numeric(x))  {
                  # mean imputation for missing numeric variables
                  mean(x, na.rm = TRUE)
                } else if(is.factor(x))  {
                  # mode imputation for missing factor variables
                  utils::tail(dimnames(sort(table(x)))[[1]], 1)
                })})
  }
  return(as.data.frame(imputed_data))
}
