#include<vector>

vector<double> cast_vector_to_double(const vector<int>& int_vector) {
    vector<double> double_vector;
    double_vector.reserve(int_vector.size()); // Reserve space to avoid reallocations

    for (int value : int_vector) {
        double_vector.push_back(static_cast<double>(value));
    }

    return double_vector;
}
