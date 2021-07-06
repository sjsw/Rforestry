#include "treeSplitting.h"
#include "forestryTree.h"
#include "DataFrame.h"
#include "RFNode.h"
#include "utils.h"
#include <RcppArmadillo.h>
#include <cmath>
#include <set>
#include <map>
#include <random>
#include <algorithm>
#include <sstream>
#include <tuple>
// [[Rcpp::plugins(cpp11)]]

double calculateRSS(
    DataFrame* trainingData,
    std::vector<size_t>* splittingSampleIndex,
    double overfitPenalty,
    std::mt19937_64& random_number_generator
) {
  // Get cross validation folds
  std::vector< std::vector< size_t > > cvFolds(10);
  if (splittingSampleIndex->size() >= 10) {
    // std::random_shuffle should NOT be used, this relies on std::rand() which is very deterministic
    std::shuffle(splittingSampleIndex->begin(), splittingSampleIndex->end(), random_number_generator);
    size_t foldIndex = 0;
    for (size_t sampleIndex : *splittingSampleIndex) {
      cvFolds.at(foldIndex).push_back(sampleIndex);
      foldIndex++;
      foldIndex = foldIndex % 10;
    }
  }

  double residualSumSquares = 0;
  size_t numFolds = cvFolds.size();
  if (splittingSampleIndex->size() < 10) {
    numFolds = 1;
  }

  for (size_t i = 0; i < numFolds; i++) {
    std::vector<size_t> trainIndex;
    std::vector<size_t> testIndex;

    if (splittingSampleIndex->size() < 10) {
      trainIndex = *splittingSampleIndex;
      testIndex = *splittingSampleIndex;
    }
    for (size_t j = 0; j < numFolds; j++) {
      if (j == i) {
        testIndex = cvFolds.at(j);
      } else {
        trainIndex.insert(trainIndex.end(), cvFolds.at(j).begin(), cvFolds.at(j).end());
      }
    }

    //Number of linear features in training data
    size_t dimension = (trainingData->getLinObsData(trainIndex[0])).size();
    arma::Mat<double> identity(dimension + 1, dimension + 1);
    identity.eye();
    arma::Mat<double> xTrain(trainIndex.size(), dimension + 1);

    //Don't penalize intercept
    identity(dimension, dimension) = 0.0;

    std::vector<double> outcomePoints;
    std::vector<double> currentObservation;

    // Contruct X and outcome vector
    for (size_t i = 0; i < trainIndex.size(); i++) {
      currentObservation = trainingData->getLinObsData((trainIndex)[i]);
      currentObservation.push_back(1.0);
      xTrain.row(i) = arma::conv_to<arma::Row<double> >::from(currentObservation);
      outcomePoints.push_back(trainingData->getOutcomePoint((trainIndex)[i]));
    }

    arma::Mat<double> y(outcomePoints.size(), 1);
    y.col(0) = arma::conv_to<arma::Col<double> >::from(outcomePoints);

    // Compute XtX + lambda * I * Y = C
    arma::Mat<double> coefficients = (xTrain.t() * xTrain +
      identity * overfitPenalty).i() * xTrain.t() * y;

    // Compute test matrix
    arma::Mat<double> xTest(testIndex.size(), dimension + 1);

    for (size_t i = 0; i < testIndex.size(); i++) {
      currentObservation = trainingData->getLinObsData((testIndex)[i]);
      currentObservation.push_back(1.0);
      xTest.row(i) = arma::conv_to<arma::Row<double> >::from(currentObservation);
    }

    arma::Mat<double> predictions = xTest * coefficients;
    for (size_t i = 0; i < predictions.size(); i++) {
      double residual = (trainingData->getOutcomePoint((testIndex)[i])) - predictions(i, 0);
      residualSumSquares += residual * residual;
    }
  }
  return residualSumSquares;
}


void updateBestSplit(
    double* bestSplitLossAll,
    double* bestSplitValueAll,
    size_t* bestSplitFeatureAll,
    size_t* bestSplitCountAll,
    double currentSplitLoss,
    double currentSplitValue,
    size_t currentFeature,
    size_t bestSplitTableIndex,
    std::mt19937_64& random_number_generator
) {

  // Update the value if a higher value has been seen
  if (currentSplitLoss > bestSplitLossAll[bestSplitTableIndex]) {
    bestSplitLossAll[bestSplitTableIndex] = currentSplitLoss;
    bestSplitFeatureAll[bestSplitTableIndex] = currentFeature;
    bestSplitValueAll[bestSplitTableIndex] = currentSplitValue;
    bestSplitCountAll[bestSplitTableIndex] = 1;
  } else {

    //If we are as good as the best split
    if (currentSplitLoss == bestSplitLossAll[bestSplitTableIndex]) {
      bestSplitCountAll[bestSplitTableIndex] =
        bestSplitCountAll[bestSplitTableIndex] + 1;

      // Only update with probability 1/nseen
      std::uniform_real_distribution<double> unif_dist;
      double tmp_random = unif_dist(random_number_generator);
      if (tmp_random * bestSplitCountAll[bestSplitTableIndex] <= 1) {
        bestSplitLossAll[bestSplitTableIndex] = currentSplitLoss;
        bestSplitFeatureAll[bestSplitTableIndex] = currentFeature;
        bestSplitValueAll[bestSplitTableIndex] = currentSplitValue;
      }
    }
  }
}

// Best split impute only additionally updates the NA direction for the splits
void updateBestSplitImpute(
    double* bestSplitLossAll,
    double* bestSplitValueAll,
    size_t* bestSplitFeatureAll,
    size_t* bestSplitCountAll,
    int* bestSplitNaDirectionAll,
    double currentSplitLoss,
    double currentSplitValue,
    size_t currentFeature,
    size_t bestSplitTableIndex,
    int currentSplitNaDirection,
    std::mt19937_64& random_number_generator
) {

  // Update the value if a higher value has been seen
  if (currentSplitLoss > bestSplitLossAll[bestSplitTableIndex]) {
    bestSplitLossAll[bestSplitTableIndex] = currentSplitLoss;
    bestSplitFeatureAll[bestSplitTableIndex] = currentFeature;
    bestSplitValueAll[bestSplitTableIndex] = currentSplitValue;
    bestSplitCountAll[bestSplitTableIndex] = 1;
    bestSplitNaDirectionAll[bestSplitTableIndex] = currentSplitNaDirection;
  } else {

    //If we are as good as the best split
    if (currentSplitLoss == bestSplitLossAll[bestSplitTableIndex]) {
      bestSplitCountAll[bestSplitTableIndex] =
        bestSplitCountAll[bestSplitTableIndex] + 1;

      // Only update with probability 1/nseen
      std::uniform_real_distribution<double> unif_dist;
      double tmp_random = unif_dist(random_number_generator);
      if (tmp_random * bestSplitCountAll[bestSplitTableIndex] <= 1) {
        bestSplitLossAll[bestSplitTableIndex] = currentSplitLoss;
        bestSplitFeatureAll[bestSplitTableIndex] = currentFeature;
        bestSplitValueAll[bestSplitTableIndex] = currentSplitValue;
        bestSplitNaDirectionAll[bestSplitTableIndex] = currentSplitNaDirection;
      }
    }
  }
}

void updateBestSplitS(
    arma::Mat<double> &bestSplitSL,
    arma::Mat<double> &bestSplitSR,
    const arma::Mat<double> &sTotal,
    DataFrame* trainingData,
    std::vector<size_t>* splittingSampleIndex,
    size_t bestSplitFeature,
    double bestSplitValue
) {
  //Get splitfeaturedata
  //sort splitindicesby splitfeature
  //while currentoutcome (getPoint(currentindex, splitfeature)) < splitValue
  //Add up outcome(i)*feat+1(i) ------ This is sL
  //sR = sTotal - sL
  //Get indexes of observations
  std::vector<size_t> splittingIndices;

  for (size_t i = 0; i < splittingSampleIndex->size(); i++) {
    splittingIndices.push_back((*splittingSampleIndex)[i]);
  }

  //Sort indices of observations ascending by currentFeature
  std::vector<double>* featureData = trainingData->getFeatureData(bestSplitFeature);

  std::sort(splittingIndices.begin(),
            splittingIndices.end(),
            [&](int fi, int si){return (*featureData)[fi] < (*featureData)[si];});

  std::vector<size_t>::iterator featIter = splittingIndices.begin();
  double currentValue = trainingData->getPoint(*featIter, bestSplitFeature);


  std::vector<double> observation;
  arma::Mat<double> crossingObservation = arma::Mat<double>(size(sTotal)).zeros();
  arma::Mat<double> sTemp = arma::Mat<double>(size(sTotal)).zeros();

  while (featIter != splittingIndices.end() &&
         currentValue < bestSplitValue
  ) {
    //Update Matriices
    observation = trainingData->getLinObsData(*featIter);
    observation.push_back(1);

    crossingObservation.col(0) =
      arma::conv_to<arma::Col<double> >::from(observation);
    crossingObservation = crossingObservation *
      trainingData->getOutcomePoint(*featIter);
    sTemp = sTemp + crossingObservation;

    ++featIter;
    currentValue = trainingData->getPoint(*featIter, bestSplitFeature);
  }

  bestSplitSL = sTemp;
  bestSplitSR = sTotal - sTemp;
}

void updateBestSplitG(
    arma::Mat<double> &bestSplitGL,
    arma::Mat<double> &bestSplitGR,
    const arma::Mat<double> &gTotal,
    DataFrame* trainingData,
    std::vector<size_t>* splittingSampleIndex,
    size_t bestSplitFeature,
    double bestSplitValue
) {

  std::vector<size_t> splittingIndices;

  for (size_t i = 0; i < splittingSampleIndex->size(); i++) {
    splittingIndices.push_back((*splittingSampleIndex)[i]);
  }

  //Sort indices of observations ascending by currentFeature
  std::vector<double>* featureData = trainingData->getFeatureData(bestSplitFeature);

  std::sort(splittingIndices.begin(),
            splittingIndices.end(),
            [&](int fi, int si){return (*featureData)[fi] < (*featureData)[si];});

  std::vector<size_t>::iterator featIter = splittingIndices.begin();
  double currentValue = trainingData->getPoint(*featIter, bestSplitFeature);


  std::vector<double> observation;
  arma::Mat<double> crossingObservation = arma::Mat<double>(size(gTotal)).zeros();
  arma::Mat<double> gTemp = arma::Mat<double>(size(gTotal)).zeros();

  while (featIter != splittingIndices.end() &&
         currentValue < bestSplitValue
  ) {
    //Update Matriices
    observation = trainingData->getLinObsData(*featIter);
    observation.push_back(1);

    crossingObservation.col(0) =
      arma::conv_to<arma::Col<double> >::from(observation);

    gTemp = gTemp + (crossingObservation * crossingObservation.t());

    ++featIter;
    currentValue = trainingData->getPoint(*featIter, bestSplitFeature);
  }

  bestSplitGL = gTemp;
  bestSplitGR = gTotal - gTemp;
}


void updateAArmadillo(
    arma::Mat<double>& a_k,
    arma::Mat<double>& new_x,
    bool leftNode
) {
  //Initilize z_K
  arma::Mat<double> z_K = a_k * new_x;

  //Update A using Sherman–Morrison formula corresponding to right or left side
  if (leftNode) {
    a_k = a_k - ((z_K) * (z_K).t()) /
      (1 + as_scalar(new_x.t() * z_K));
  } else {
    a_k = a_k + ((z_K) * (z_K).t()) /
      (1 - as_scalar(new_x.t() * z_K));
  }
}

void updateSkArmadillo(
    arma::Mat<double>& s_k,
    arma::Mat<double>& next,
    double next_y,
    bool left
) {
  if (left) {
    s_k = s_k + (next_y * (next));
  } else {
    s_k = s_k - (next_y * (next));
  }
}

double computeRSSArmadillo(
    arma::Mat<double>& A_r,
    arma::Mat<double>& A_l,
    arma::Mat<double>& S_r,
    arma::Mat<double>& S_l,
    arma::Mat<double>& G_r,
    arma::Mat<double>& G_l
) {
  return (as_scalar((S_l.t() * A_l) * (G_l * (A_l * S_l))) +
          as_scalar((S_r.t() * A_r) * (G_r * (A_r * S_r))) -
          as_scalar(2.0 * S_l.t() * (A_l * S_l)) -
          as_scalar(2.0 * S_r.t() * (A_r * S_r)));
}



void updateRSSComponents(
    DataFrame* trainingData,
    size_t nextIndex,
    arma::Mat<double>& aLeft,
    arma::Mat<double>& aRight,
    arma::Mat<double>& sLeft,
    arma::Mat<double>& sRight,
    arma::Mat<double>& gLeft,
    arma::Mat<double>& gRight,
    arma::Mat<double>& crossingObservation,
    arma::Mat<double>& obOuter
) {
  //Get observation that will cross the partition
  std::vector<double> newLeftObservation =
    trainingData->getLinObsData(nextIndex);

  newLeftObservation.push_back(1.0);

  crossingObservation.col(0) =
    arma::conv_to<arma::Col<double> >::from(newLeftObservation);

  double crossingOutcome = trainingData->getOutcomePoint(nextIndex);

  //Use to update RSS components
  updateSkArmadillo(sLeft, crossingObservation, crossingOutcome, true);
  updateSkArmadillo(sRight, crossingObservation, crossingOutcome, false);

  obOuter = crossingObservation * crossingObservation.t();
  gLeft = gLeft + obOuter;
  gRight = gRight - obOuter;

  updateAArmadillo(aLeft, crossingObservation, true);
  updateAArmadillo(aRight, crossingObservation, false);
}

void initializeRSSComponents(
    DataFrame* trainingData,
    size_t index,
    size_t numLinearFeatures,
    double overfitPenalty,
    const arma::Mat<double>& gTotal,
    const arma::Mat<double>& sTotal,
    arma::Mat<double>& aLeft,
    arma::Mat<double>& aRight,
    arma::Mat<double>& sLeft,
    arma::Mat<double>& sRight,
    arma::Mat<double>& gLeft,
    arma::Mat<double>& gRight,
    arma::Mat<double>& crossingObservation
) {
  //Initialize sLeft
  sLeft = trainingData->getOutcomePoint(index) *crossingObservation;

  sRight = sTotal - sLeft;

  //Initialize gLeft
  gLeft = crossingObservation * (crossingObservation.t());

  gRight = gTotal - gLeft;
  //Initialize sRight, gRight

  arma::Mat<double> identity(numLinearFeatures + 1,
                             numLinearFeatures + 1);
  identity.eye();

  //Don't penalize intercept
  identity(numLinearFeatures, numLinearFeatures) = 0.0;
  identity = overfitPenalty * identity;

  //Initialize aLeft
  aLeft = (gLeft + identity).i();

  //Initialize aRight
  aRight = (gRight + identity).i();
}

double calcMuBarVar(
    // Calculates proxy for MSE of potential split
    double leftSum, size_t leftCount,
    double totalSum, size_t totalCount
) {
  double parentMean = totalSum/totalCount;
  double leftMeanCentered  = leftSum/leftCount - parentMean;
  double rightMeanCentered = (totalSum - leftSum)/(totalCount - leftCount) - parentMean;
  double muBarVarSum = leftCount * leftMeanCentered * leftMeanCentered +
    (totalCount - leftCount) * rightMeanCentered * rightMeanCentered;
  return muBarVarSum/totalCount;
}


void findBestSplitRidgeCategorical(
    std::vector<size_t>* averagingSampleIndex,
    std::vector<size_t>* splittingSampleIndex,
    size_t bestSplitTableIndex,
    size_t currentFeature,
    double* bestSplitLossAll,
    double* bestSplitValueAll,
    size_t* bestSplitFeatureAll,
    size_t* bestSplitCountAll,
    DataFrame* trainingData,
    size_t splitNodeSize,
    size_t averageNodeSize,
    std::mt19937_64& random_number_generator,
    double overfitPenalty,
    std::shared_ptr< arma::Mat<double> > gtotal,
    std::shared_ptr< arma::Mat<double> > stotal
) {
  /* Put all categories in a set
   * aggregate G_k matrices to put in left node when splitting
   * aggregate S_k and G_k matrices at each step
   *
   * linearly iterate through averaging indices adding count to total set count
   *
   * linearly iterate thought splitting indices and add G_k to the matrix mapped
   * to each index, then put in the all categories set
   *
   * Left is aggregated, right is total - aggregated
   * subtract and feed to RSS calculator for each partition
   * call updateBestSplitRidge with correct G_k matrices
   */

  // Set to hold all different categories
  std::set<double> all_categories;
  std::vector<double> temp;

  // temp matrices for RSS components
  arma::Mat<double> gRightTemp(size((*gtotal)));
  arma::Mat<double> sRightTemp(size((*stotal)));
  arma::Mat<double> aRightTemp(size((*gtotal)));
  arma::Mat<double> aLeftTemp(size((*gtotal)));
  arma::Mat<double> crossingObservation(size((*stotal)));
  arma::Mat<double> identity(size((*gtotal)));

  identity.eye();
  identity(identity.n_rows-1, identity.n_cols-1) = 0.0;
  size_t splitTotalCount = 0;
  size_t averageTotalCount = 0;

  // Create map to track the count and RSS components
  std::map<double, size_t> splittingCategoryCount;
  std::map<double, size_t> averagingCategoryCount;
  std::map<double, arma::Mat<double> > gMatrices;
  std::map<double, arma::Mat<double> > sMatrices;

  for (size_t j=0; j<averagingSampleIndex->size(); j++) {
    all_categories.insert(
      (*trainingData).getPoint((*averagingSampleIndex)[j], currentFeature)
    );
    averageTotalCount++;
  }

  for (size_t j=0; j<splittingSampleIndex->size(); j++) {
    all_categories.insert(
      (*trainingData).getPoint((*splittingSampleIndex)[j], currentFeature)
    );
    splitTotalCount++;
  }

  for (
      std::set<double>::iterator it=all_categories.begin();
      it != all_categories.end();
      ++it
  ) {
    splittingCategoryCount[*it] = 0;
    averagingCategoryCount[*it] = 0;
    gMatrices[*it] = arma::Mat<double>(size(*gtotal)).zeros();
    sMatrices[*it] = arma::Mat<double>(size(*stotal)).zeros();
  }

  // Put all matrices in map
  for (size_t j = 0; j<splittingSampleIndex->size(); j++) {
    // Add each observation to correct matrix in map
    double currentCategory = trainingData->getPoint((*splittingSampleIndex)[j],
                                                    currentFeature);
    double currentOutcome =
      trainingData->getOutcomePoint((*splittingSampleIndex)[j]);

    temp = trainingData->getLinObsData((*splittingSampleIndex)[j]);
    temp.push_back(1);
    crossingObservation.col(0) = arma::conv_to<arma::Col<double> >::from(temp);

    updateSkArmadillo(sMatrices[currentCategory],
                      crossingObservation,
                      currentOutcome,
                      true);

    gMatrices[currentCategory] = gMatrices[currentCategory]
    +crossingObservation * crossingObservation.t();
    splittingCategoryCount[currentCategory]++;
  }

  for (size_t j=0; j<(*averagingSampleIndex).size(); j++) {
    double currentCategory = (*trainingData).
    getPoint((*averagingSampleIndex)[j], currentFeature);
    averagingCategoryCount[currentCategory]++;
  }

  // Evaluate possible splits using associated RSS components
  for (
      std::set<double>::iterator it=all_categories.begin();
      it != all_categories.end();
      ++it
  ) {
    // Check leaf size at least nodesize
    if (
        std::min(
          splittingCategoryCount[*it],
                                splitTotalCount - splittingCategoryCount[*it]
        ) < splitNodeSize ||
          std::min(
            averagingCategoryCount[*it],
                                  averageTotalCount - averagingCategoryCount[*it]
          ) < averageNodeSize
    ) {
      continue;
    }
    gRightTemp = (*gtotal) - gMatrices[*it];
    sRightTemp = (*stotal) - sMatrices[*it];

    aRightTemp = (gRightTemp + overfitPenalty * identity).i();
    aLeftTemp = (gMatrices[*it] + overfitPenalty * identity).i();

    double currentSplitLoss = computeRSSArmadillo(aRightTemp,
                                                  aLeftTemp,
                                                  sRightTemp,
                                                  sMatrices[*it],
                                                           gRightTemp,
                                                           gMatrices[*it]);

    updateBestSplit(
      bestSplitLossAll,
      bestSplitValueAll,
      bestSplitFeatureAll,
      bestSplitCountAll,
      -currentSplitLoss,
      (double) *it,
      currentFeature,
      bestSplitTableIndex,
      random_number_generator
    );
  }
}

void findBestSplitValueCategorical(
    std::vector<size_t>* averagingSampleIndex,
    std::vector<size_t>* splittingSampleIndex,
    size_t bestSplitTableIndex,
    size_t currentFeature,
    double* bestSplitLossAll,
    double* bestSplitValueAll,
    size_t* bestSplitFeatureAll,
    size_t* bestSplitCountAll,
    DataFrame* trainingData,
    size_t splitNodeSize,
    size_t averageNodeSize,
    std::mt19937_64& random_number_generator,
    size_t maxObs
) {

  // Count total number of observations for different categories
  std::set<double> all_categories;
  double splitTotalSum = 0;
  size_t splitTotalCount = 0;
  size_t averageTotalCount = 0;

  //EDITED
  //Move indices to vectors so we can downsample if needed
  std::vector<size_t> splittingIndices;
  std::vector<size_t> averagingIndices;

  for (size_t y = 0; y < (*splittingSampleIndex).size(); y++) {
    splittingIndices.push_back((*splittingSampleIndex)[y]);
  }

  for (size_t y = 0; y < (*averagingSampleIndex).size(); y++) {
    averagingIndices.push_back((*averagingSampleIndex)[y]);
  }


  //If maxObs is smaller, randomly downsample
  if (maxObs < (*splittingSampleIndex).size()) {
    std::vector<size_t> newSplittingIndices;
    std::vector<size_t> newAveragingIndices;

    std::shuffle(splittingIndices.begin(), splittingIndices.end(),
                 random_number_generator);
    std::shuffle(averagingIndices.begin(), averagingIndices.end(),
                 random_number_generator);

    for (size_t q = 0; q < maxObs; q++) {
      newSplittingIndices.push_back(splittingIndices[q]);
      newAveragingIndices.push_back(averagingIndices[q]);
    }

    std::swap(newSplittingIndices, splittingIndices);
    std::swap(newAveragingIndices, averagingIndices);
  }

  for (size_t j=0; j<splittingIndices.size(); j++) {
    all_categories.insert(
      (*trainingData).getPoint(splittingIndices[j], currentFeature)
    );
    splitTotalSum +=
      (*trainingData).getOutcomePoint(splittingIndices[j]);
    splitTotalCount++;
  }

  for (size_t j=0; j<averagingIndices.size(); j++) {
    all_categories.insert(
      (*trainingData).getPoint(averagingIndices[j], currentFeature)
    );
    averageTotalCount++;
  }

  // Create map to track the count and sum of y squares
  std::map<double, size_t> splittingCategoryCount;
  std::map<double, size_t> averagingCategoryCount;
  std::map<double, double> splittingCategoryYSum;

  for (
      std::set<double>::iterator it=all_categories.begin();
      it != all_categories.end();
      ++it
  ) {
    splittingCategoryCount[*it] = 0;
    averagingCategoryCount[*it] = 0;
    splittingCategoryYSum[*it] = 0;
  }

  for (size_t j=0; j<(*splittingSampleIndex).size(); j++) {
    double currentXValue = (*trainingData).
    getPoint((*splittingSampleIndex)[j], currentFeature);
    double currentYValue = (*trainingData).
    getOutcomePoint((*splittingSampleIndex)[j]);
    splittingCategoryCount[currentXValue] += 1;
    splittingCategoryYSum[currentXValue] += currentYValue;
  }

  for (size_t j=0; j<(*averagingSampleIndex).size(); j++) {
    double currentXValue = (*trainingData).
    getPoint((*averagingSampleIndex)[j], currentFeature);
    averagingCategoryCount[currentXValue] += 1;
  }

  // Go through the sums and determine the best partition
  for (
      std::set<double>::iterator it=all_categories.begin();
      it != all_categories.end();
      ++it
  ) {
    // Check leaf size at least nodesize
    if (
        std::min(
          splittingCategoryCount[*it],
                                splitTotalCount - splittingCategoryCount[*it]
        ) < splitNodeSize ||
        std::min(
          averagingCategoryCount[*it],
                                averageTotalCount - averagingCategoryCount[*it]
        ) < averageNodeSize
    ) {
      continue;
    }

    double currentSplitLoss = calcMuBarVar(
      splittingCategoryYSum[*it],
                           splittingCategoryCount[*it],
                                                 splitTotalSum,
                                                 splitTotalCount
    );

    updateBestSplit(
      bestSplitLossAll,
      bestSplitValueAll,
      bestSplitFeatureAll,
      bestSplitCountAll,
      currentSplitLoss,
      *it,
      currentFeature,
      bestSplitTableIndex,
      random_number_generator
    );
  }
}

void findBestSplitRidge(
    std::vector<size_t>* averagingSampleIndex,
    std::vector<size_t>* splittingSampleIndex,
    size_t bestSplitTableIndex,
    size_t currentFeature,
    double* bestSplitLossAll,
    double* bestSplitValueAll,
    size_t* bestSplitFeatureAll,
    size_t* bestSplitCountAll,
    DataFrame* trainingData,
    size_t splitNodeSize,
    size_t averageNodeSize,
    std::mt19937_64& random_number_generator,
    bool splitMiddle,
    size_t maxObs,
    double overfitPenalty,
    std::shared_ptr< arma::Mat<double> > gtotal,
    std::shared_ptr< arma::Mat<double> > stotal
) {

  //Get indexes of observations
  std::vector<size_t> splittingIndexes;
  std::vector<size_t> averagingIndexes;

  for (size_t i = 0; i < splittingSampleIndex->size(); i++) {
    splittingIndexes.push_back((*splittingSampleIndex)[i]);
  }

  for (size_t j = 0; j < averagingSampleIndex->size(); j++) {
    averagingIndexes.push_back((*averagingSampleIndex)[j]);
  }

  //Sort indexes of observations ascending by currentFeature
  std::vector<double>* featureData = trainingData->getFeatureData(currentFeature);

  sort(splittingIndexes.begin(),
       splittingIndexes.end(),
       [&](int fi, int si){return (*featureData)[fi] < (*featureData)[si];});

  sort(averagingIndexes.begin(),
       averagingIndexes.end(),
       [&](int fi, int si){return (*featureData)[fi] < (*featureData)[si];});

  size_t splitLeftCount = 0;
  size_t averageLeftCount = 0;
  size_t splitTotalCount = splittingIndexes.size();
  size_t averageTotalCount = averagingIndexes.size();

  std::vector<size_t>::iterator splitIter = splittingIndexes.begin();
  std::vector<size_t>::iterator averageIter = averagingIndexes.begin();
  /* Increment splitIter because we have initialized RSS components with
   * observation from splitIter.begin(), so we need to avoid duplicate 1st obs
   */


  //Now begin splitting
  size_t currentIndex;

  /* Need at least one splitOb to evaluate RSS */
  currentIndex = (*splitIter);
  ++splitIter;
  splitLeftCount++;

  /* Move appropriate averagingObs to left */

  while (
      averageIter < averagingIndexes.end() && (
          trainingData->getPoint((*averageIter), currentFeature) <=
            trainingData->getPoint(currentIndex, currentFeature))
  ) {
    ++averageIter;
    averageLeftCount++;
  }

  double currentValue = trainingData->getPoint(currentIndex, currentFeature);

  size_t newIndex;
  size_t numLinearFeatures;
  bool oneDistinctValue = true;

  //Initialize RSS components
  //TODO: think about completely duplicate observations

  std::vector<double> firstOb = trainingData->getLinObsData(currentIndex);

  numLinearFeatures = firstOb.size();
  firstOb.push_back(1.0);

  //Initialize crossingObs for body of loop
  arma::Mat<double> crossingObservation(firstOb.size(),
                                        1);

  arma::Mat<double> obOuter(numLinearFeatures + 1,
                            numLinearFeatures + 1);

  crossingObservation.col(0) = arma::conv_to<arma::Col<double> >::from(firstOb);

  arma::Mat<double> aLeft(numLinearFeatures + 1, numLinearFeatures + 1),
  aRight(numLinearFeatures + 1, numLinearFeatures + 1),
  gLeft(numLinearFeatures + 1, numLinearFeatures + 1),
  gRight(numLinearFeatures + 1, numLinearFeatures + 1),
  sLeft(numLinearFeatures + 1, 1),
  sRight(numLinearFeatures + 1, 1);

  initializeRSSComponents(
    trainingData,
    currentIndex,
    numLinearFeatures,
    overfitPenalty,
    (*gtotal),
    (*stotal),
    aLeft,
    aRight,
    sLeft,
    sRight,
    gLeft,
    gRight,
    crossingObservation
  );

  while (
      splitIter < splittingIndexes.end() ||
        averageIter < averagingIndexes.end()
  ) {

    currentValue = trainingData->getPoint(currentIndex, currentFeature);
    //Move iterators forward
    while (
        splitIter < splittingIndexes.end() &&
          trainingData->getPoint((*splitIter), currentFeature) <= currentValue
    ) {
      //UPDATE RSS pieces with current splitIter index
      updateRSSComponents(
        trainingData,
        (*splitIter),
        aLeft,
        aRight,
        sLeft,
        sRight,
        gLeft,
        gRight,
        crossingObservation,
        obOuter
      );

      splitLeftCount++;
      ++splitIter;
    }

    while (
        averageIter < averagingIndexes.end() &&
          trainingData->getPoint((*averageIter), currentFeature) <=
          currentValue
    ) {
      averageLeftCount++;
      ++averageIter;
    }

    //Test if we only have one feature value to be considered
    if (oneDistinctValue) {
      oneDistinctValue = false;
      if (
          splitIter == splittingIndexes.end() &&
            averageIter == averagingIndexes.end()
      ) {
        break;
      }
    }

    //Set newIndex to index iterator with the minimum currentFeature value
    if (
        splitIter == splittingIndexes.end() &&
          averageIter == averagingIndexes.end()
    ) {
      break;
    } else if (
        splitIter == splittingIndexes.end()
    ) {
      /* Can't pass down matrix if we split past last splitting index */
      break;
    } else if (
        averageIter == averagingIndexes.end()
    ) {
      newIndex = (*splitIter);
    } else if (
        trainingData->getPoint((*averageIter), currentFeature) <
          trainingData->getPoint((*splitIter), currentFeature)
    ) {
      newIndex = (*averageIter);
    } else {
      newIndex = (*splitIter);
    }

    //Check if split would create a node too small
    if (
        std::min(
          splitLeftCount,
          splitTotalCount - splitLeftCount
        ) < splitNodeSize ||
          std::min(
            averageLeftCount,
            averageTotalCount - averageLeftCount
          ) < averageNodeSize
    ) {
      currentIndex = newIndex;
      continue;
    }

    //Sum of RSS's of models fit on left and right partitions
    double currentRSS = computeRSSArmadillo(aRight,
                                            aLeft,
                                            sRight,
                                            sLeft,
                                            gRight,
                                            gLeft);

    double currentSplitValue;

    double featureValue = trainingData->getPoint(currentIndex, currentFeature);

    double newFeatureValue = trainingData->getPoint(newIndex, currentFeature);

    if (splitMiddle) {
      currentSplitValue = (featureValue + newFeatureValue) / 2.0;
    } else {
      std::uniform_real_distribution<double> unif_dist;
      double tmp_random = unif_dist(random_number_generator) *
        (newFeatureValue - featureValue);
      double epsilon_lower = std::nextafter(featureValue, newFeatureValue);
      double epsilon_upper = std::nextafter(newFeatureValue, featureValue);
      currentSplitValue = tmp_random + featureValue;
      if (currentSplitValue > epsilon_upper) {
        currentSplitValue = epsilon_upper;
      }
      if (currentSplitValue < epsilon_lower) {
        currentSplitValue = epsilon_lower;
      }
    }
    //Rcpp::Rcout << currentRSS << " " << currentSplitValue << "\n";
    updateBestSplit(
      bestSplitLossAll,
      bestSplitValueAll,
      bestSplitFeatureAll,
      bestSplitCountAll,
      -currentRSS,
      currentSplitValue,
      currentFeature,
      bestSplitTableIndex,
      random_number_generator
    );
    currentIndex = newIndex;
  }
}


void findBestSplitValueNonCategorical(
    std::vector<size_t>* averagingSampleIndex,
    std::vector<size_t>* splittingSampleIndex,
    size_t bestSplitTableIndex,
    size_t currentFeature,
    double* bestSplitLossAll,
    double* bestSplitValueAll,
    size_t* bestSplitFeatureAll,
    size_t* bestSplitCountAll,
    DataFrame* trainingData,
    size_t splitNodeSize,
    size_t averageNodeSize,
    std::mt19937_64& random_number_generator,
    bool splitMiddle,
    size_t maxObs,
    bool monotone_splits,
    monotonic_info monotone_details
) {

  // Create specific vectors to holddata
  typedef std::tuple<double,double> dataPair;
  std::vector<dataPair> splittingData;
  std::vector<dataPair> averagingData;
  double splitTotalSum = 0;
  double avgTotalSum = 0;

  for (size_t j=0; j<(*splittingSampleIndex).size(); j++){
    // Retrieve the current feature value
    double tmpFeatureValue = (*trainingData).
    getPoint((*splittingSampleIndex)[j], currentFeature);
    double tmpOutcomeValue = (*trainingData).
    getOutcomePoint((*splittingSampleIndex)[j]);
    splitTotalSum += tmpOutcomeValue;

    // Adding data to the internal data vector (Note: R index)
    splittingData.push_back(
      std::make_tuple(
        tmpFeatureValue,
        tmpOutcomeValue
      )
    );
  }



  for (size_t j=0; j<(*averagingSampleIndex).size(); j++){
    // Retrieve the current feature value
    double tmpFeatureValue = (*trainingData).
    getPoint((*averagingSampleIndex)[j], currentFeature);
    double tmpOutcomeValue = (*trainingData).
    getOutcomePoint((*averagingSampleIndex)[j]);
    avgTotalSum += tmpOutcomeValue;

    // Adding data to the internal data vector (Note: R index)
    averagingData.push_back(
      std::make_tuple(
        tmpFeatureValue,
        tmpOutcomeValue
      )
    );
  }
  // If there are more than maxSplittingObs, randomly downsample maxObs samples
  if (maxObs < splittingData.size()) {

    std::vector<dataPair> newSplittingData;
    std::vector<dataPair> newAveragingData;

    std::shuffle(splittingData.begin(), splittingData.end(),
                 random_number_generator);
    std::shuffle(averagingData.begin(), averagingData.end(),
                 random_number_generator);

    for (size_t q = 0; q < maxObs; q++) {
      newSplittingData.push_back(splittingData[q]);
      newAveragingData.push_back(averagingData[q]);
    }

    std::swap(newSplittingData, splittingData);
    std::swap(newAveragingData, averagingData);

  }

  // Sort both splitting and averaging dataset
  sort(
    splittingData.begin(),
    splittingData.end(),
    [](const dataPair &lhs, const dataPair &rhs) {
      return std::get<0>(lhs) < std::get<0>(rhs);
    }
  );
  sort(
    averagingData.begin(),
    averagingData.end(),
    [](const dataPair &lhs, const dataPair &rhs) {
      return std::get<0>(lhs) < std::get<0>(rhs);
    }
  );

  size_t splitLeftPartitionCount = 0;
  size_t averageLeftPartitionCount = 0;
  size_t splitTotalCount = splittingData.size();
  size_t averageTotalCount = averagingData.size();

  double splitLeftPartitionRunningSum = 0;
  double avgLeftPartitionRunningSum = 0;

  std::vector<dataPair>::iterator splittingDataIter = splittingData.begin();
  std::vector<dataPair>::iterator averagingDataIter = averagingData.begin();

  // Initialize the split value to be minimum of first value in two datasets
  double featureValue = std::min(
    std::get<0>(*splittingDataIter),
    std::get<0>(*averagingDataIter)
  );

  double newFeatureValue;
  bool oneValueDistinctFlag = true;

  while (
      splittingDataIter < splittingData.end() ||
        averagingDataIter < averagingData.end()
  ){

    // Exhaust all current feature value in both datasets as partitioning
    while (
        splittingDataIter < splittingData.end() &&
          std::get<0>(*splittingDataIter) == featureValue
    ) {
      splitLeftPartitionCount++;
      splitLeftPartitionRunningSum += std::get<1>(*splittingDataIter);
      splittingDataIter++;
    }

    while (
        averagingDataIter < averagingData.end() &&
          std::get<0>(*averagingDataIter) == featureValue
    ) {
      averageLeftPartitionCount++;
      avgLeftPartitionRunningSum += std::get<1>(*averagingDataIter);
      averagingDataIter++;
    }

    // Test if the all the values for the feature are the same, then proceed
    if (oneValueDistinctFlag) {
      oneValueDistinctFlag = false;
      if (
          splittingDataIter == splittingData.end() &&
            averagingDataIter == averagingData.end()
      ) {
        break;
      }
    }

    // Make partitions on the current feature and value in both splitting
    // and averaging dataset. `averageLeftPartitionCount` and
    // `splitLeftPartitionCount` already did the partition after we sort the
    // array.
    // Get new feature value
    if (
        splittingDataIter == splittingData.end() &&
          averagingDataIter == averagingData.end()
    ) {
      break;
    } else if (splittingDataIter == splittingData.end()) {
      newFeatureValue = std::get<0>(*averagingDataIter);
    } else if (averagingDataIter == averagingData.end()) {
      newFeatureValue = std::get<0>(*splittingDataIter);
    } else {
      newFeatureValue = std::min(
        std::get<0>(*splittingDataIter),
        std::get<0>(*averagingDataIter)
      );
    }

    // Check leaf size at least nodesize
    if (
        std::min(
          splitLeftPartitionCount,
          splitTotalCount - splitLeftPartitionCount
        ) < splitNodeSize ||
          std::min(
            averageLeftPartitionCount,
            averageTotalCount - averageLeftPartitionCount
          ) < averageNodeSize
    ) {
      // Update the oldFeature value before proceeding
      featureValue = newFeatureValue;
      continue;
    }

    // If we are using monotonic constraints, we need to work out whether
    // the monotone constraints will reject a split
    if (monotone_splits) {
      bool keepMonotoneSplit = acceptMonotoneSplit(monotone_details,
                                                   currentFeature,
                                                   splitLeftPartitionRunningSum / splitLeftPartitionCount,
                                                   (splitTotalSum - splitLeftPartitionRunningSum)
                                                     / (splitTotalCount - splitLeftPartitionCount));

      bool avgKeepMonotoneSplit = true;
      // If monotoneAvg, we also need to check the monotonicity of the avg set
      if (monotone_details.monotoneAvg) {
        avgKeepMonotoneSplit = acceptMonotoneSplit(monotone_details,
                                                   currentFeature,
                                                   avgLeftPartitionRunningSum / averageLeftPartitionCount,
                                                   (avgTotalSum - avgLeftPartitionRunningSum)
                                                     / (averageTotalCount - averageLeftPartitionCount));

      }

      if (!(keepMonotoneSplit && avgKeepMonotoneSplit)) {
        // Update the oldFeature value before proceeding
        featureValue = newFeatureValue;
        continue;
      }
    }

    // Calculate the variance of the splitting
    double currentSplitLoss = calcMuBarVar(
      splitLeftPartitionRunningSum,
      splitLeftPartitionCount,
      splitTotalSum,
      splitTotalCount);


    double currentSplitValue;
    if (splitMiddle) {
      currentSplitValue = (newFeatureValue + featureValue) / 2.0;
    } else {
      std::uniform_real_distribution<double> unif_dist;
      double tmp_random = unif_dist(random_number_generator) *
        (newFeatureValue - featureValue);
      double epsilon_lower = std::nextafter(featureValue, newFeatureValue);
      double epsilon_upper = std::nextafter(newFeatureValue, featureValue);
      currentSplitValue = tmp_random + featureValue;
      if (currentSplitValue > epsilon_upper) {
        currentSplitValue = epsilon_upper;
      }
      if (currentSplitValue < epsilon_lower) {
        currentSplitValue = epsilon_lower;
      }
    }

    updateBestSplit(
      bestSplitLossAll,
      bestSplitValueAll,
      bestSplitFeatureAll,
      bestSplitCountAll,
      currentSplitLoss,
      currentSplitValue,
      currentFeature,
      bestSplitTableIndex,
      random_number_generator
    );

    // Update the old feature value
    featureValue = newFeatureValue;
  }
}

void findBestSplitImpute(
    std::vector<size_t>* averagingSampleIndex,
    std::vector<size_t>* splittingSampleIndex,
    size_t bestSplitTableIndex,
    size_t currentFeature,
    double* bestSplitLossAll,
    double* bestSplitValueAll,
    size_t* bestSplitFeatureAll,
    size_t* bestSplitCountAll,
    int* bestSplitNaDirectionAll,
    DataFrame* trainingData,
    size_t splitNodeSize,
    size_t averageNodeSize,
    std::mt19937_64& random_number_generator,
    bool splitMiddle,
    size_t maxObs,
    bool monotone_splits,
    monotonic_info monotone_details
) {

  // Create specific vectors to holddata
  typedef std::tuple<double,double> dataPair;
  std::vector<dataPair> splittingData;
  std::vector<dataPair> averagingData;

  //Create vector to hold missing data
  typedef std::tuple<size_t, double> naPair;
  std::vector<naPair> missingSplit;
  std::vector<naPair> missingAvg;


  double splitTotalSum = 0;
  double avgTotalSum = 0;
  double naTotalSum = 0;
  double naAvgTotalSum = 0;
  size_t naAvgTotalCount = 0;
  size_t naSplTotalCount = 0;


  for (size_t j=0; j<(*splittingSampleIndex).size(); j++) {
    // Retrieve the current feature value
    double tmpFeatureValue = (*trainingData).
    getPoint((*splittingSampleIndex)[j], currentFeature);
    double tmpOutcomeValue = (*trainingData).
    getOutcomePoint((*splittingSampleIndex)[j]);

    // If feature data is missing, push back to missingData vector
    if (std::isnan(tmpFeatureValue)) {
      naTotalSum += tmpOutcomeValue;
      naSplTotalCount++;

      missingSplit.push_back(
        std::make_tuple(
          (*splittingSampleIndex)[j],
          tmpOutcomeValue
        )
      );
    } else {
      splitTotalSum += tmpOutcomeValue;

      // Adding data to the internal data vector
      splittingData.push_back(
        std::make_tuple(
          tmpFeatureValue,
          tmpOutcomeValue
        )
      );
    }
  }

  for (size_t j=0; j<(*averagingSampleIndex).size(); j++){
    // Retrieve the current feature value
    double tmpFeatureValue = (*trainingData).
    getPoint((*averagingSampleIndex)[j], currentFeature);
    double tmpOutcomeValue = (*trainingData).
    getOutcomePoint((*averagingSampleIndex)[j]);

    if (std::isnan(tmpFeatureValue)) {
      naAvgTotalSum += tmpOutcomeValue;
      naAvgTotalCount++;

      missingAvg.push_back(
        std::make_tuple(
          (*averagingSampleIndex)[j],
          tmpOutcomeValue
        )
      );
    } else {
      // Adding data to the internal data vector
      avgTotalSum += tmpOutcomeValue;

      averagingData.push_back(
        std::make_tuple(
          tmpFeatureValue,
          tmpOutcomeValue
        )
      );
    }
  }

  // return if we have no data
  if ( (splittingData.size() < 1) || (averagingData.size() < 1) )
  {
    return;
  }


  // If there are more than maxSplittingObs, randomly downsample maxObs samples
  if (maxObs < splittingData.size()) {

    std::vector<dataPair> newSplittingData;
    std::vector<dataPair> newAveragingData;

    std::shuffle(splittingData.begin(), splittingData.end(),
                 random_number_generator);
    std::shuffle(averagingData.begin(), averagingData.end(),
                 random_number_generator);

    for (size_t q = 0; q < maxObs; q++) {
      newSplittingData.push_back(splittingData[q]);
      newAveragingData.push_back(averagingData[q]);
    }

    std::swap(newSplittingData, splittingData);
    std::swap(newAveragingData, averagingData);
  }

  // Sort both splitting and averaging dataset
  sort(
    splittingData.begin(),
    splittingData.end(),
    [](const dataPair &lhs, const dataPair &rhs) {
      return std::get<0>(lhs) < std::get<0>(rhs);
    }
  );

  sort(
    averagingData.begin(),
    averagingData.end(),
    [](const dataPair &lhs, const dataPair &rhs) {
      return std::get<0>(lhs) < std::get<0>(rhs);
    }
  );

  size_t splitLeftPartitionCount = 0;
  size_t averageLeftPartitionCount = 0;
  size_t splitTotalCount = splittingData.size();
  size_t averageTotalCount = averagingData.size();

  double splitLeftPartitionRunningSum = 0;
  double avgLeftPartitionRunningSum = 0;

  std::vector<dataPair>::iterator splittingDataIter = splittingData.begin();
  std::vector<dataPair>::iterator averagingDataIter = averagingData.begin();

  // Initialize the split value to be minimum of first value in two datsets
  double featureValue = std::min(
    std::get<0>(*splittingDataIter),
    std::get<0>(*averagingDataIter)
  );

  double newFeatureValue;
  bool oneValueDistinctFlag = true;

  while (
      splittingDataIter < splittingData.end() ||
        averagingDataIter < averagingData.end()
  ) {

    // Exhaust all current feature value in both dataset as partitioning
    while (
        splittingDataIter < splittingData.end() &&
          std::get<0>(*splittingDataIter) == featureValue
    ) {
      splitLeftPartitionCount++;
      splitLeftPartitionRunningSum += std::get<1>(*splittingDataIter);
      splittingDataIter++;
    }

    while (
        averagingDataIter < averagingData.end() &&
          std::get<0>(*averagingDataIter) == featureValue
    ) {
      averageLeftPartitionCount++;
      avgLeftPartitionRunningSum += std::get<1>(*averagingDataIter);
      averagingDataIter++;
    }

    // Test if the all the values for the feature are the same, then proceed
    if (oneValueDistinctFlag) {
      oneValueDistinctFlag = false;
      if (
          splittingDataIter == splittingData.end()
      ) {
        break;
      }
    }

    // Make partitions on the current feature and value in both splitting
    // and averaging dataset. `averageLeftPartitionCount` and
    // `splitLeftPartitionCount` already did the partition after we sort the
    // array.
    // Get new feature value
    if (
        splittingDataIter == splittingData.end() &&
          averagingDataIter == averagingData.end()
    ) {
      break;
    } else if (splittingDataIter == splittingData.end()) {
      newFeatureValue = std::get<0>(*averagingDataIter);
    } else if (averagingDataIter == averagingData.end()) {
      newFeatureValue = std::get<0>(*splittingDataIter);
    } else {
      newFeatureValue = std::min(
        std::get<0>(*splittingDataIter),
        std::get<0>(*averagingDataIter)
      );
    }

    // Check leaf size at least nodesize
    if (
        std::min(
          splitLeftPartitionCount,
          splitTotalCount - splitLeftPartitionCount
        ) < splitNodeSize ||
          std::min(
            averageLeftPartitionCount,
            averageTotalCount - averageLeftPartitionCount
          ) < averageNodeSize
    ) {
      // Update the oldFeature value before proceeding
      featureValue = newFeatureValue;
      continue;
    }

    double currentSplitValue;
    if (splitMiddle) {
      currentSplitValue = (newFeatureValue + featureValue) / 2.0;
    } else {
      std::uniform_real_distribution<double> unif_dist;
      double tmp_random = unif_dist(random_number_generator) *
        (newFeatureValue - featureValue);
      double epsilon_lower = std::nextafter(featureValue, newFeatureValue);
      double epsilon_upper = std::nextafter(newFeatureValue, featureValue);
      currentSplitValue = tmp_random + featureValue;
      if (currentSplitValue > epsilon_upper) {
        currentSplitValue = epsilon_upper;
      }
      if (currentSplitValue < epsilon_lower) {
        currentSplitValue = epsilon_lower;
      }
    }

    // For monotonicity with missing data, we need to to check both left and right
    // handling of NA's respects monotonicity
    bool avgKeepMonotoneSplitLeft = true;
    bool avgKeepMonotoneSplitRight = true;
    bool keepMonotoneSplitLeft = true;
    bool keepMonotoneSplitRight = true;

    if (monotone_splits) {
      // First check left
      keepMonotoneSplitLeft =
        acceptMonotoneSplit(monotone_details,
                            currentFeature,
                            (splitLeftPartitionRunningSum + naTotalSum) /
                              (splitLeftPartitionCount + naSplTotalCount),
                              (splitTotalSum - splitLeftPartitionRunningSum + naTotalSum)
                              / (splitTotalCount - splitLeftPartitionCount + naSplTotalCount));

      keepMonotoneSplitRight =
        acceptMonotoneSplit(monotone_details,
                            currentFeature,
                            (splitLeftPartitionRunningSum) /
                              (splitLeftPartitionCount),
                              (splitTotalSum - splitLeftPartitionRunningSum + naTotalSum)
                              / (splitTotalCount - splitLeftPartitionCount + naSplTotalCount));


      // If monotoneAvg, we also need to check the monotonicity of the avg set
      if (monotone_details.monotoneAvg) {
        avgKeepMonotoneSplitLeft =
          acceptMonotoneSplit(monotone_details,
                              currentFeature,
                              (avgLeftPartitionRunningSum + naAvgTotalSum) /
                                (averageLeftPartitionCount + naAvgTotalCount),
                              (avgTotalSum - avgLeftPartitionRunningSum + naAvgTotalSum)
                                / (averageTotalCount - averageLeftPartitionCount + naAvgTotalCount));
        avgKeepMonotoneSplitRight =
          acceptMonotoneSplit(monotone_details,
                              currentFeature,
                              (avgLeftPartitionRunningSum) /
                                (averageLeftPartitionCount),
                              (avgTotalSum - avgLeftPartitionRunningSum + naAvgTotalSum)
                                / (averageTotalCount - averageLeftPartitionCount + naAvgTotalCount));
      }
    }


    // Calculate variance of the splitting using updated partition means and counts

    // Calculate MuBarVar if we send all NA's to the left
    if (keepMonotoneSplitLeft && avgKeepMonotoneSplitLeft) {
      double currentSplitLossLeft = calcMuBarVar(
        (splitLeftPartitionRunningSum + naTotalSum),
        (splitLeftPartitionCount + naSplTotalCount),
        (splitTotalSum + naTotalSum),
        (splitTotalCount + naSplTotalCount));

      updateBestSplitImpute(
        bestSplitLossAll,
        bestSplitValueAll,
        bestSplitFeatureAll,
        bestSplitCountAll,
        bestSplitNaDirectionAll,
        currentSplitLossLeft,
        currentSplitValue,
        currentFeature,
        bestSplitTableIndex,
        -1,
        random_number_generator
      );
    }

    // Calculate MuBarVar if we send all NA's to the right
    if (keepMonotoneSplitRight && avgKeepMonotoneSplitRight) {
      double currentSplitLossRight = calcMuBarVar(
        splitLeftPartitionRunningSum,
        splitLeftPartitionCount,
        (splitTotalSum + naTotalSum),
        (splitTotalCount + naSplTotalCount));

      updateBestSplitImpute(
        bestSplitLossAll,
        bestSplitValueAll,
        bestSplitFeatureAll,
        bestSplitCountAll,
        bestSplitNaDirectionAll,
        currentSplitLossRight,
        currentSplitValue,
        currentFeature,
        bestSplitTableIndex,
        1,
        random_number_generator
      );
    }

    // Update the old feature value
    featureValue = newFeatureValue;
  }
}

void findBestSplitImputeCategorical(
    std::vector<size_t>* averagingSampleIndex,
    std::vector<size_t>* splittingSampleIndex,
    size_t bestSplitTableIndex,
    size_t currentFeature,
    double* bestSplitLossAll,
    double* bestSplitValueAll,
    size_t* bestSplitFeatureAll,
    size_t* bestSplitCountAll,
    int* bestSplitNaDirectionAll,
    DataFrame* trainingData,
    size_t splitNodeSize,
    size_t averageNodeSize,
    std::mt19937_64& random_number_generator,
    size_t maxObs
) {

  // Count total number of observations for different categories
  std::set<double> all_categories;
  double splitTotalSum = 0;
  double naTotalSum = 0;
  size_t splitTotalCount = 0;
  size_t averageTotalCount = 0;
  size_t totalNaCount = 0;

  typedef std::tuple<size_t, double> naPair;
  std::vector<naPair> missingSplit;
  std::vector<naPair> missingAvg;


  //EDITED
  //Move indices to vectors so we can downsample if needed
  std::vector<size_t> splittingIndices;
  std::vector<size_t> averagingIndices;

  for (size_t y = 0; y < (*splittingSampleIndex).size(); y++) {
    splittingIndices.push_back((*splittingSampleIndex)[y]);
  }

  for (size_t y = 0; y < (*averagingSampleIndex).size(); y++) {
    averagingIndices.push_back((*averagingSampleIndex)[y]);
  }


  //If maxObs is smaller, randomly downsample
  if (maxObs < (*splittingSampleIndex).size()) {
    std::vector<size_t> newSplittingIndices;
    std::vector<size_t> newAveragingIndices;

    std::shuffle(splittingIndices.begin(), splittingIndices.end(),
                 random_number_generator);
    std::shuffle(averagingIndices.begin(), averagingIndices.end(),
                 random_number_generator);

    for (size_t q = 0; q < maxObs; q++) {
      newSplittingIndices.push_back(splittingIndices[q]);
      newAveragingIndices.push_back(averagingIndices[q]);
    }

    std::swap(newSplittingIndices, splittingIndices);
    std::swap(newAveragingIndices, averagingIndices);
  }

  // Keep track of all categories and counts in both splitting and Avg datasets
  for (size_t j=0; j<splittingIndices.size(); j++) {
    all_categories.insert(
      (*trainingData).getPoint(splittingIndices[j], currentFeature)
    );
    splitTotalSum +=
      (*trainingData).getOutcomePoint(splittingIndices[j]);
    splitTotalCount++;
  }
  for (size_t j=0; j<averagingIndices.size(); j++) {
    all_categories.insert(
      (*trainingData).getPoint(averagingIndices[j], currentFeature)
    );
    averageTotalCount++;
  }


  // Create map to track the count and sum of y squares
  std::map<double, size_t> splittingCategoryCount;
  std::map<double, size_t> averagingCategoryCount;
  std::map<double, double> splittingCategoryYSum;

  for (
      std::set<double>::iterator it=all_categories.begin();
      it != all_categories.end();
      ++it
  ) {
    splittingCategoryCount[*it] = 0;
    averagingCategoryCount[*it] = 0;
    splittingCategoryYSum[*it] = 0;
  }

  for (size_t j=0; j<(*splittingSampleIndex).size(); j++) {
    double currentXValue = (*trainingData).
    getPoint((*splittingSampleIndex)[j], currentFeature);
    double currentYValue = (*trainingData).
    getOutcomePoint((*splittingSampleIndex)[j]);

    if (std::isnan(currentXValue)) {
      totalNaCount++;
      naTotalSum += currentYValue;
      missingSplit.push_back(
        std::make_tuple(
          (*splittingSampleIndex)[j],
                                 currentYValue
        )
      );

    } else {
      splittingCategoryCount[currentXValue] += 1;
      splittingCategoryYSum[currentXValue] += currentYValue;
    }
  }

  for (size_t j=0; j<(*averagingSampleIndex).size(); j++) {
    double currentXValue = (*trainingData).
    getPoint((*averagingSampleIndex)[j], currentFeature);

    if (!std::isnan(currentXValue)) {
      averagingCategoryCount[currentXValue] += 1;
    }
  }

  // Go through the sums and determine the best partition
  for (
      std::set<double>::iterator it=all_categories.begin();
      it != all_categories.end();
      ++it
  ) {
    // Check leaf size at least nodesize
    if (
        std::min(
          splittingCategoryCount[*it],
                                splitTotalCount - splittingCategoryCount[*it]
        ) < splitNodeSize ||
          std::min(
            averagingCategoryCount[*it],
                                  averageTotalCount - averagingCategoryCount[*it]
          ) < averageNodeSize
    ) {
      continue;
    }

    // Now filter NA values by outcome value which are closest to mean of each side of partition
    // Update left/right mean and count by sum and number of NA's and give new splitloss
    double currentSplitLossLeft =
      calcMuBarVar(
        (splittingCategoryYSum[*it] + naTotalSum),
        (splittingCategoryCount[*it] + totalNaCount),
        (splitTotalSum + naTotalSum),
        (splitTotalCount + totalNaCount)
      );

    updateBestSplitImpute(
      bestSplitLossAll,
      bestSplitValueAll,
      bestSplitFeatureAll,
      bestSplitCountAll,
      bestSplitNaDirectionAll,
      currentSplitLossLeft,
      *it,
      currentFeature,
      bestSplitTableIndex,
      -1,
      random_number_generator
    );

    double currentSplitLossRight =
      calcMuBarVar(
        splittingCategoryYSum[*it],
        splittingCategoryCount[*it],
        (splitTotalSum + naTotalSum),
        (splitTotalCount + totalNaCount)
      );

    updateBestSplitImpute(
      bestSplitLossAll,
      bestSplitValueAll,
      bestSplitFeatureAll,
      bestSplitCountAll,
      bestSplitNaDirectionAll,
      currentSplitLossRight,
      *it,
      currentFeature,
      bestSplitTableIndex,
      1,
      random_number_generator
    );

  }
}


void findBestSplitSymmetric(
    std::vector<size_t>* averagingSampleIndex,
    std::vector<size_t>* splittingSampleIndex,
    size_t bestSplitTableIndex,
    size_t currentFeature,
    double* bestSplitLossAll,
    double* bestSplitValueAll,
    size_t* bestSplitFeatureAll,
    size_t* bestSplitCountAll,
    DataFrame* trainingData,
    size_t splitNodeSize,
    size_t averageNodeSize,
    std::mt19937_64& random_number_generator,
    bool splitMiddle,
    size_t maxObs,
    bool monotone_splits,
    monotonic_info monotone_details
) {
  // In order to model an outcome which is a symmetric function of the inputs
  // we devise a way to select nonlinear splits which (when given appropriate
  // symmetric outcomes) minimizes the mse of the three resulting partitions.

  // Pseudo code of the splitting algorithm is as follows:

  // For epsilon > 0, we start with symmetric splits at +- epsilon, and
  // consider consecutive split points at each +- feature value, working
  // from smallest in absolute value to greatest.

  // At each possible pairs of splits, we have three partitions, the inner,
  // the left, and the right. At each point, we take the mean outcome of the
  // inner outcomes, and +- n_L * |M_L| + n_R * |M_R| / (n_L + n_R) the weighted
  // magnitude of the left and right means, with signs based on the appropriate
  // ordering.

  // The loss of these constant aggregations is calculated for each split point,
  // and the split is rejected if it does not respect monotonicity.

  // The size of these three partitions are checked against the nodesizeStrict
  // and the pseudo outcomes are also taken for predictions.

  typedef std::tuple<double,double,double> dataPair;
  std::vector<dataPair> splittingData;
  std::vector<dataPair> averagingData;

  double leftRunningSum = 0;
  double rightRunningSum = 0;
  double midRunningSum = 0;

  for (size_t j=0; j<(*splittingSampleIndex).size(); j++) {
    // Retrieve the current feature value
    double tmpFeatureValue = (*trainingData).
    getPoint((*splittingSampleIndex)[j], currentFeature);
    double tmpOutcomeValue = (*trainingData).
    getOutcomePoint((*splittingSampleIndex)[j]);

    // Adding data to the internal data vector (Note: R index)
    splittingData.push_back(
      std::make_tuple(
        tmpFeatureValue,
        std::abs(tmpFeatureValue),
        tmpOutcomeValue
      )
    );
  }


  for (size_t j=0; j<(*averagingSampleIndex).size(); j++) {
    // Retrieve the current feature value
    double tmpFeatureValue = (*trainingData).
    getPoint((*averagingSampleIndex)[j], currentFeature);
    double tmpOutcomeValue = (*trainingData).
    getOutcomePoint((*averagingSampleIndex)[j]);

    // Adding data to the internal data vector (Note: R index)
    averagingData.push_back(
      std::make_tuple(
        tmpFeatureValue,
        std::abs(tmpFeatureValue),
        tmpOutcomeValue
      )
    );
  }

  // Now sort possible splitting points by absolute feature value
  sort(
    splittingData.begin(),
    splittingData.end(),
    [](const dataPair &lhs, const dataPair &rhs) {
      return std::get<1>(lhs) < std::get<1>(rhs);
    }
  );

  size_t nLeft = 0;
  size_t nRight = 0;
  size_t nMid = 0;

  size_t nAvgLeft = 0;
  size_t nAvgRight = 0;
  size_t nAvgMid = 0;

  double midWeight;
  double leftWeight;
  double rightWeight;

  double newFeatureValue;
  bool oneValueDistinctFlag = true;

  // Now iterate through split points, and initialize lhs and rhs sums
  for (const auto& dataPoint : splittingData) {
    if (std::get<0>(dataPoint) > 0) {
      rightRunningSum += std::get<2>(dataPoint);
      nLeft++;
    } else {
      leftRunningSum += std::get<2>(dataPoint);
      nRight++;
    }
  }

  for (const auto& dataPoint : averagingData) {
    if (std::get<0>(dataPoint) > 0) {
      nAvgLeft++;
    } else {
      nAvgRight++;
    }
  }

  // Now work on determining the optimal split
  std::vector<dataPair>::iterator splittingDataIter = splittingData.begin();
  std::vector<dataPair>::iterator averagingDataIter = averagingData.begin();

  // Initialize the split value to be minimum of first value in two datasets
  double featureValue = std::min(
    std::get<1>(*splittingDataIter),
    std::get<1>(*averagingDataIter)
  );

  while (
      splittingDataIter < splittingData.end() ||
        averagingDataIter < averagingData.end()
  ) {

    // Exhaust all current feature value in both datasets as partitioning
    while (
        splittingDataIter < splittingData.end() &&
          std::get<1>(*splittingDataIter) == featureValue
    ) {
      // We check if the current value is in the left or right partition
      if (std::get<0>(*splittingDataIter) > 0) {
        nMid++;
        nRight--;
        rightRunningSum -= std::get<2>(*splittingDataIter);
        midRunningSum += std::get<2>(*splittingDataIter);
      } else {
        nMid++;
        nLeft--;
        leftRunningSum -= std::get<2>(*splittingDataIter);
        midRunningSum += std::get<2>(*splittingDataIter);
      }
    }

    while (
        averagingDataIter < averagingData.end() &&
          std::get<1>(*averagingDataIter) == featureValue
    ) {
      // We check if the current value is in the left or right partition
      if (std::get<0>(*averagingDataIter) > 0) {
        nAvgMid++;
        nAvgRight--;
      } else {
        nAvgMid++;
        nAvgLeft--;
      }
    }

    // Test if the all the values for the feature are the same, then proceed
    if (oneValueDistinctFlag) {
      oneValueDistinctFlag = false;
      if (
          splittingDataIter == splittingData.end() &&
            averagingDataIter == averagingData.end()
      ) {
        break;
      }
    }

    // Update the splitting value to the next feature value with the smallest absolute value
    if (
        splittingDataIter == splittingData.end() &&
          averagingDataIter == averagingData.end()
    ) {
      break;
    } else if (splittingDataIter == splittingData.end()) {
      newFeatureValue = std::get<1>(*averagingDataIter);
    } else if (averagingDataIter == averagingData.end()) {
      newFeatureValue = std::get<1>(*splittingDataIter);
    } else {
      newFeatureValue = std::min(
        std::get<1>(*splittingDataIter),
        std::get<1>(*averagingDataIter)
      );
    }

    // Check nodesize for all three partitions
    if (
        std::min(
          nLeft,
          std::min(nRight,nMid)
        ) < splitNodeSize ||
          std::min(
            nAvgLeft,
            std::min(nAvgRight,nAvgMid)
          ) < averageNodeSize
    ) {
      featureValue = newFeatureValue;
      continue;
    }

    // Get the appropriate partition weights given the means and counts
    updatePartitionWeights(leftRunningSum/(double) nLeft,
                           midRunningSum / (double) nMid,
                           rightRunningSum/(double) nRight,
                           nLeft,
                           nRight,
                           nMid,
                           leftWeight,
                           rightWeight,
                           midWeight);

    // Calculate the variance of the splitting
    double currentSplitLoss = calcSymmetricLoss(
      leftRunningSum,
      midRunningSum,
      rightRunningSum,
      nLeft,
      nRight,
      nMid,
      leftWeight,
      rightWeight,
      midWeight);


    double currentSplitValue;
    if (splitMiddle) {
      currentSplitValue = (newFeatureValue + featureValue) / 2.0;
    } else {
      std::uniform_real_distribution<double> unif_dist;
      double tmp_random = unif_dist(random_number_generator) *
        (newFeatureValue - featureValue);
      double epsilon_lower = std::nextafter(featureValue, newFeatureValue);
      double epsilon_upper = std::nextafter(newFeatureValue, featureValue);
      currentSplitValue = tmp_random + featureValue;
      if (currentSplitValue > epsilon_upper) {
        currentSplitValue = epsilon_upper;
      }
      if (currentSplitValue < epsilon_lower) {
        currentSplitValue = epsilon_lower;
      }
    }

    updateBestSplit(
      bestSplitLossAll,
      bestSplitValueAll,
      bestSplitFeatureAll,
      bestSplitCountAll,
      -currentSplitLoss, // Standard RF split loss we want to maximize due to
      currentSplitValue, // the splitting trick, here we want to minimize, so we
      currentFeature,    // flip the sign when picking the best.
      bestSplitTableIndex,
      random_number_generator
    );

    // Update the old feature value
    featureValue = newFeatureValue;
  }
}

double calcSymmetricLoss(
    double leftSum,
    double midSum,
    double rightSum,
    size_t nLeft,
    size_t nRight,
    size_t nMid,
    double leftWeight,
    double rightWeight,
    double midWeight
) {
  return(((double) nLeft)*leftWeight*leftWeight +
         ((double) nMid)*midWeight*midWeight +
         ((double) nRight)*rightWeight*rightWeight -
         2*(leftWeight*leftSum + midWeight*midSum + rightWeight*rightSum));
}

void updatePartitionWeights(
    double leftMean,
    double midMean,
    double rightMean,
    size_t nLeft,
    size_t nRight,
    size_t nMid,
    double &leftWeight,
    double &rightWeight,
    double &midWeight
) {
  // Update the partition weights given new partition means and sizes in order
  // to ensure symmetric weights

  midWeight = midMean;

  double average_diff = std::abs(midWeight - (((double) nLeft) * leftMean + ((double) nRight) * rightMean)/
                                   (((double) nLeft) + ((double) nRight))
                                );

  if (leftMean < rightMean)
    {
    leftWeight = midWeight - average_diff;
    rightWeight = midWeight + average_diff;
    }
  else
    {
    leftWeight = midWeight + average_diff;
    rightWeight = midWeight - average_diff;
    }
}

void determineBestSplit(
    size_t &bestSplitFeature,
    double &bestSplitValue,
    double &bestSplitLoss,
    int &bestSplitNaDir,
    size_t mtry,
    double* bestSplitLossAll,
    double* bestSplitValueAll,
    size_t* bestSplitFeatureAll,
    size_t* bestSplitCountAll,
    int* bestSplitNaDirectionAll,
    std::mt19937_64& random_number_generator
) {

  // Get the best split values among all features
  double bestSplitLoss_ = -std::numeric_limits<double>::infinity();
  std::vector<size_t> bestFeatures;

  for (size_t i=0; i<mtry; i++) {
    if (bestSplitLossAll[i] > bestSplitLoss_) {
      bestSplitLoss_ = bestSplitLossAll[i];
    }
  }

  for (size_t i=0; i<mtry; i++) {
    if (bestSplitLossAll[i] == bestSplitLoss_) {
      for (size_t j=0; j<bestSplitCountAll[i]; j++) {
        bestFeatures.push_back(i);
      }
    }
  }

  // If we found a feasible splitting point
  if (bestFeatures.size() > 0) {

    // If there are multiple best features, sample one according to their
    // frequency of occurence
    std::uniform_int_distribution<size_t> unif_dist(
        0, bestFeatures.size() - 1
    );
    size_t tmp_random = unif_dist(random_number_generator);
    size_t bestFeatureIndex = bestFeatures.at(tmp_random);
    // Return the best splitFeature and splitValue
    bestSplitFeature = bestSplitFeatureAll[bestFeatureIndex];
    bestSplitValue = bestSplitValueAll[bestFeatureIndex];
    bestSplitNaDir = bestSplitNaDirectionAll[bestFeatureIndex];
    bestSplitLoss = bestSplitLoss_;
  } else {
    // If none of the features are possible, return NA
    bestSplitFeature = std::numeric_limits<size_t>::quiet_NaN();
    bestSplitValue = std::numeric_limits<double>::quiet_NaN();
    bestSplitLoss = std::numeric_limits<double>::quiet_NaN();
  }

}

bool acceptMonotoneSplit(
    monotonic_info &monotone_details,
    size_t currentFeature,
    double leftPartitionMean,
    double rightPartitionMean
) {
  // If we have the uncle mean equal to infinity, then we enforce a simple
  // monotone split without worrying about the uncle bounds
  int monotone_direction = monotone_details.monotonic_constraints[currentFeature];
  double upper_bound = monotone_details.upper_bound;
  double lower_bound = monotone_details.lower_bound;

  // This is not right. I should check the split is correctly monotonic and then
  // check that neither node violates the upper and lower bounds

  // Monotone increasing
  if ((monotone_direction == 1) && (leftPartitionMean > rightPartitionMean)) {
    return false;
  } else if ((monotone_direction == -1) && (rightPartitionMean > leftPartitionMean)) {
    // Monotone decreasing
    return false;
  } else if ((monotone_direction == 1) && (rightPartitionMean > upper_bound)) {
    return false;
  } else if ((monotone_direction == 1) && (leftPartitionMean < lower_bound)) {
    return false;
  } else if ((monotone_direction == -1) && (rightPartitionMean < lower_bound)) {
    return false;
  } else if ((monotone_direction == -1) && (leftPartitionMean > upper_bound)) {
    return false;
  } else if (monotone_direction == 0) {
    if (std::min(leftPartitionMean, rightPartitionMean) < lower_bound ||
        std::max(leftPartitionMean, rightPartitionMean) > upper_bound) {
      return false;
    } else {
      return true;
    }
  } else {
    return true;
  }
}

double calculateMonotonicBound(
    double node_mean,
    monotonic_info& monotone_details
) {
  if (node_mean < monotone_details.lower_bound) {
    return monotone_details.lower_bound;
  } else if (node_mean > monotone_details.upper_bound) {
    return monotone_details.upper_bound;
  } else {
    return node_mean;
  }
}
