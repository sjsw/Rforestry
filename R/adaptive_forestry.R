#' @include forestry.R

setClass(
  Class = "adaptiveForestry",
  slots = list(
    first.forest = "forestry",
    second.forest = "forestry",
    ntree.first = "numeric",
    ntree.second = "numeric",
    split.props = "numeric"
  )
)
# --- Adaptive Forestry --------------------------------------------------------
#' adaptiveForestry
#' @name adaptiveForestry-forestry
#' @title forestry with adaptive featureWeights
#' @rdname adaptiveForestry-forestry
#' @description This is an experimental function where we run forestry in two
#'   stages, first estimating the feature weights by calculating the relative
#'   splitting proportions of each feature using a small forest, and then
#'   growing a much bigger forest using the
#'   first forest splitting proportions as the featureWeights in the second forest.
#' @inheritParams forestry
#' @param ntree.first The number of trees to grow in the first forest when
#'   trying to determine which features are important.
#' @param ntree.second The number of features to use in the second stage when
#'   we grow a second forest using the weights of the first stage.
#' @return Two forestry objects, the first forest, and the adaptive forest,
#'   as well as the splitting proportions used to grow the second forest.
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
#' rf <- adaptiveForestry(x = x_train,
#'                         y = y_train,
#'                         ntree.first = 25,
#'                         ntree.second = 500,
#'                         nthread = 2)
#' predict(rf@second.forest, x_test)
#'
#' @export
adaptiveForestry <- function(x,
                              y,
                              ntree = 500,
                              ntree.first = 25,
                              ntree.second = 500,
                              replace = TRUE,
                              sampsize = if (replace)
                                nrow(x)
                              else
                                ceiling(.632 * nrow(x)),
                              sample.fraction = NULL,
                              mtry = max(floor(ncol(x) / 3), 1),
                              nodesizeSpl = 5,
                              nodesizeAvg = 5,
                              nodesizeStrictSpl = 1,
                              nodesizeStrictAvg = 1,
                              minSplitGain = 0,
                              maxDepth = round(nrow(x) / 2) + 1,
                              interactionDepth = maxDepth,
                              interactionVariables = numeric(0),
                              featureWeights = NULL,
                              deepFeatureWeights = NULL,
                              observationWeights = NULL,
                              splitratio = 1,
                              OOBhonest = FALSE,
                              seed = as.integer(runif(1) * 1000),
                              verbose = FALSE,
                              nthread = 0,
                              splitrule = "variance",
                              middleSplit = FALSE,
                              maxObs = length(y),
                              linear = FALSE,
                              linFeats = 0:(ncol(x)-1),
                              monotonicConstraints = rep(0, ncol(x)),
                              monotoneAvg = FALSE,
                              overfitPenalty = 1,
                              doubleTree = FALSE,
                              reuseforestry = NULL,
                              savable = TRUE,
                              saveable = TRUE) {



  first.forest = forestry(x = x,
                          y = y,
                          ntree = ntree.first,
                          replace = replace,
                          sampsize = sampsize,
                          sample.fraction = sample.fraction,
                          mtry = mtry,
                          nodesizeSpl = nodesizeSpl,
                          nodesizeAvg = nodesizeAvg,
                          nodesizeStrictSpl = nodesizeStrictSpl,
                          nodesizeStrictAvg = nodesizeStrictAvg,
                          minSplitGain = minSplitGain,
                          maxDepth = maxDepth,
                          interactionDepth = interactionDepth,
                          interactionVariables = interactionVariables,
                          featureWeights = featureWeights,
                          deepFeatureWeights = deepFeatureWeights,
                          observationWeights = observationWeights,
                          splitratio = splitratio,
                          OOBhonest = OOBhonest,
                          seed = seed,
                          verbose = verbose,
                          nthread = nthread,
                          splitrule = splitrule,
                          middleSplit = middleSplit,
                          maxObs = maxObs,
                          linear = linear,
                          linFeats = linFeats,
                          monotonicConstraints = monotonicConstraints,
                          monotoneAvg = monotoneAvg,
                          overfitPenalty = overfitPenalty,
                          doubleTree = doubleTree,
                          reuseforestry = reuseforestry,
                          savable = savable,
                          saveable = saveable)

  splitting_props <- getSplitProps(first.forest)

  second.forest <- forestry(x = x,
                            y = y,
                            ntree = ntree.first,
                            replace = replace,
                            sampsize = sampsize,
                            sample.fraction = sample.fraction,
                            mtry = mtry,
                            nodesizeSpl = nodesizeSpl,
                            nodesizeAvg = nodesizeAvg,
                            nodesizeStrictSpl = nodesizeStrictSpl,
                            nodesizeStrictAvg = nodesizeStrictAvg,
                            minSplitGain = minSplitGain,
                            maxDepth = maxDepth,
                            interactionDepth = interactionDepth,
                            interactionVariables = interactionVariables,
                            featureWeights = unname(splitting_props),
                            # Give the splitting proportions as the feature weights
                            deepFeatureWeights = deepFeatureWeights,
                            observationWeights = observationWeights,
                            splitratio = splitratio,
                            OOBhonest = OOBhonest,
                            seed = seed,
                            verbose = verbose,
                            nthread = nthread,
                            splitrule = splitrule,
                            middleSplit = middleSplit,
                            maxObs = maxObs,
                            linear = linear,
                            linFeats = linFeats,
                            monotonicConstraints = monotonicConstraints,
                            monotoneAvg = monotoneAvg,
                            overfitPenalty = overfitPenalty,
                            doubleTree = doubleTree,
                            reuseforestry = reuseforestry,
                            savable = savable,
                            saveable = saveable)
  return(
    new(
      "adaptiveForestry",
      first.forest=first.forest,
      second.forest=second.forest,
      ntree.first=ntree.first,
      ntree.second=ntree.second,
      split.props=splitting_props
    )
  )
}

# -- Predict Method for adaptiveForestry ---------------------------------------
#' predict-adaptiveForestry
#' @name predict-adaptiveForestry
#' @rdname predict-adaptiveForestry
#' @description Return the prediction from the forest.
#' @param object An `adaptiveForestry` object.
#' @param newdata A data frame of testing predictors.
#' @param aggregation How the individual tree predictions are aggregated:
#'   `average` returns the mean of all trees in the forest; `weightMatrix`
#'   returns a list consisting of "weightMatrix", the adaptive nearest neighbor
#'   weights used to construct the predictions; "terminalNodes", a matrix where
#'   the ith entry of the jth column is the index of the leaf node to which the
#'   ith observation is assigned in the jth tree; and "sparse", a matrix
#'   where the ith entry in the jth column is 1 if the ith observation in
#'   feature.new is assigned to the jth leaf and 0 otherwise. In each tree the
#'   leaves are indexed using a depth first ordering, and, in the "sparse"
#'   representation, the first leaf in the second tree has column index one more than
#'   the number of leaves in the first tree and so on. So, for example, if the
#'   first tree has 5 leaves, the sixth column of the "sparse" matrix corresponds
#'   to the first leaf in the second tree.
#' @param seed random seed
#' @param nthread The number of threads with which to run the predictions with.
#'   This will default to the number of threads with which the forest was trained
#'   with.
#' @param exact This specifies whether the forest predictions should be aggregated
#'   in a reproducible ordering. Due to the non-associativity of floating point
#'   addition, when we predict in parallel, predictions will be aggregated in
#'   varied orders as different threads finish at different times.
#'   By default, exact is TRUE unless N > 100,000 or a custom aggregation
#'   function is used.
#' @param weighting This should be a number between 0 and 1 indicating the
#'   weight with which to use the predictions of the two forests. This
#'   specifically specifies the weight given to the second.forest object. The
#'   predictions are given by weighting * predict(object@second.forest) +
#'   (1-weighting) * predict(object@first.forest).
#'   Defaults to NULL, and in this case, weighting = ntree.second / (ntree.first + ntree.second).
#' @param ... additional arguments.
#' @return A vector of predicted responses.
#' @export
predict.adaptiveForestry <- function(object,
                                     newdata,
                                     aggregation = "average",
                                     seed = as.integer(runif(1) * 10000),
                                     nthread = 0,
                                     exact = NULL,
                                     weighting = NULL,
                                     ...) {

  if (is.null(weighting)) {
    weighting <- object@ntree.second / (object@ntree.first + object@ntree.second)
  } else if (weighting > 1 || weighting < 0) {
    stop("weighting must be between 0 and 1")
  }

  # Get first forest predictions
  p.first <- predict(object = object@first.forest,
                     newdata = newdata,
                     aggregation = "average",
                     seed = seed,
                     nthread = nthread,
                     exact = exact)

  # Get second forest predictions
  p.second <- predict(object = object@second.forest,
                      newdata = newdata,
                      aggregation = "average",
                      seed = seed,
                      nthread = nthread,
                      exact = exact)

  return(weighting*p.first + (1-weighting)*p.second)


}
