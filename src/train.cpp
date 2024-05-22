#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>

using namespace mlpack;

int main() {
    // Load dataset.csv
    data::DatasetInfo info;
    arma::mat dataset;
    data::Load("dataset.csv", dataset, true, false, info);

    // Features and target variables
    arma::mat X = dataset.cols(1, 3); // Assuming columns 0, 1, 2 are '나이(Age)', '키(Height)', '몸무게(Weight)'
    arma::rowvec y_left = dataset.col(4); // Assuming column 3 is 'LPressureVal'
    arma::rowvec y_right = dataset.col(5); // Assuming column 4 is 'RPressureVal'

    // Split data into training and testing sets
    arma::mat X_train, X_test;
    arma::rowvec y_left_train, y_left_test, y_right_train, y_right_test;
    mlpack::data::Split(X, y_left, y_right, X_train, X_test, y_left_train, y_left_test, y_right_train, y_right_test, 0.2);

    // Train Random Forest Regressor for left foot pressure
    mlpack::regression::RandomForest<> model_left(X_train, y_left_train, 1, 10);
    
    // Train Random Forest Regressor for right foot pressure
    mlpack::regression::RandomForest<> model_right(X_train, y_right_train, 1, 10);

    // Save the trained models
    model_left.Train(X_train, y_left_train);
    model_right.Train(X_train, y_right_train);

    mlpack::regression::RandomForest<>::Save("AI_model_left.xml");
    mlpack::regression::RandomForest<>::Save("AI_model_right.xml");

    return 0;
}