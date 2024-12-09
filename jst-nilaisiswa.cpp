#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

// Fungsi aktivasi (sigmoid)
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Turunan fungsi sigmoid untuk backpropagation
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

int main() {
    // Data input: nilai jam pelajaran, praktik, kerjasama, UTS, dan UAS (normalized 0-1)
    vector<vector<double>> inputs = {
        {0.25, 0.02, 0.05, 0.0, 0.23},  // Data siswa 1
        {0.51, 0.4, 0.15, 0.05, 0.49},  // Data siswa 2
        {0.32, 0.3, 0.1, 0.05, 0.29},  // Data siswa 3
    };

    // Target output: nilai rapor siswa (normalized 0-1)
    vector<double> outputs = {0.21, 0.47, 0.27}; 

    // Inisialisasi bobot secara acak
    vector<double> weights = {0.5, 0.5, 0.5, 0.5, 0.5}; 
    double bias = 0.5;                        // Bias awal

    double learning_rate = 0.1; // Kecepatan belajar model
    int epochs = 10000;         // Jumlah iterasi

    // Proses training menggunakan backpropagation
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_error = 0.0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            // Forward pass
            double weighted_sum = bias;
            for (size_t j = 0; j < inputs[i].size(); ++j) {
                weighted_sum += inputs[i][j] * weights[j];
            }

            double predicted_output = sigmoid(weighted_sum);

            // Hitung error (selisih output sebenarnya dengan prediksi)
            double error = outputs[i] - predicted_output;
            total_error += error * error;

            // Backpropagation (perbaikan bobot dan bias)
            double delta = error * sigmoid_derivative(predicted_output);

            for (size_t j = 0; j < weights.size(); ++j) {
                weights[j] += learning_rate * delta * inputs[i][j];
            }

            bias += learning_rate * delta;
        }

        // Cetak error pada beberapa iterasi
        if (epoch % 1000 == 0) {
            cout << "Epoch " << epoch << ", Total Error: " << total_error << endl;
        }
    }

    // Prediksi setelah training
    cout << "\nPrediksi Nilai Rapor:" << endl;
    for (size_t i = 0; i < inputs.size(); ++i) {
        double weighted_sum = bias;
        for (size_t j = 0; j < inputs[i].size(); ++j) {
            weighted_sum += inputs[i][j] * weights[j];
        }

        double predicted_output = sigmoid(weighted_sum);
        cout << "Siswa " << i + 1 << ": " << predicted_output * 10 << " (skala 0-10)" << endl;
    }

    return 0;
}