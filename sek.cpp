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
    // Data input: tugas, ujian, partisipasi (normalized 0-1)
    vector<vector<double>> inputs = {
        {0.5, 0.0, 2.3},  // Data siswa 1
        {1.5, 0.5, 4.9},  // Data siswa 2
        {1.0, 0.5, 2.9},  // Data siswa 3
    };

    // Target output: nilai rapor siswa (normalized 0-1)
    vector<double> outputs = {0.6, 0.9, 0.8}; 

    // Inisialisasi bobot secara acak
    vector<double> weights = {0.5, 0.5, 0.5}; 
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
