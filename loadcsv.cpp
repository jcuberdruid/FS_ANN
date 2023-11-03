#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

const int IMAGE_SIZE = 784;

bool load_mnist_csv(const string &filename, vector<int> &labels, vector<vector<int>> &images) {
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return false;
    }

    string line;
    getline(file, line); // Skip the header line

    int line_count = 0;
    while (getline(file, line)) {
        stringstream linestream(line);
        string value;
        int label;
        int pixel;
        int value_count = 0;

        if (!getline(linestream, value, ',')) {
            cerr << "Error reading label on line " << line_count << endl;
            return false;
        }

        try {
            label = stoi(value);
        } catch (const invalid_argument& ia) {
            cerr << "Invalid argument: " << ia.what() << " at line " << line_count << " value: " << value << endl;
            return false;
        }

        labels.push_back(label);
        vector<int> image;
        image.reserve(IMAGE_SIZE);

        while (getline(linestream, value, ',')) {
            try {
                pixel = stoi(value);
            } catch (const invalid_argument& ia) {
                cerr << "Invalid argument: " << ia.what() << " at line " << line_count << " value: " << value << endl;
                return false;
            }
            image.push_back(pixel);
            value_count++;
        }

        if (value_count != IMAGE_SIZE) {
            cerr << "Incorrect number of values on line " << line_count << endl;
            return false;
        }

        images.push_back(image);
        line_count++;
    }

    file.close();
    return true;
}

vector<vector<int>> one_hot_encode(const vector<int>& input, int num_classes) {
    vector<vector<int>> encoded(input.size(), vector<int>(num_classes, 0));

    for (size_t i = 0; i < input.size(); ++i) {
        if (input[i] >= 0 && input[i] < num_classes) {
            encoded[i][input[i]] = 1;
        } else {
            cerr << "Value out of range: " << input[i] << endl;
            // Handle the error according to your needs.
            // Here, we just print an error message.
        }
    }

    return encoded;
}


/*
//test main
int main() {
    vector<int> labels;
    vector<vector<int>> images;

    if (load_mnist_csv("mnist_train.csv", labels, images)) {
        cout << "Loaded " << images.size() << " training examples." << endl;
    } else {
        cerr << "Failed to load training data." << endl;
        return 1;
    }
    cout << labels[0] << endl;
    return 0;
}
*/
