#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <thread>
#include <mutex>

// Random generator per thread
double generateGaussianNoise(double mean, double stddev, std::mt19937& generator) {
    std::normal_distribution<double> distribution(mean, stddev);
    return distribution(generator);
}

double callOptionPayoff(double S, double K) {
    return std::max(S - K, 0.0);
}

double putOptionPayoff(double S, double K) {
    return std::max(K - S, 0.0);
}

void monteCarloWorker(double S0, double K, double r, double sigma, double T, int simulationsPerThread, bool isCallOption, double& payoffSum, std::mutex& mtx) {
    std::random_device rd;
    std::mt19937 generator(rd());
    double localPayoffSum = 0.0;

    for (int i = 0; i < simulationsPerThread; ++i) {
        double Z = generateGaussianNoise(0.0, 1.0, generator);
        double ST = S0 * std::exp((r - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * Z);
        double payoff = isCallOption ? callOptionPayoff(ST, K) : putOptionPayoff(ST, K);
        localPayoffSum += payoff;
    }

    std::lock_guard<std::mutex> lock(mtx);
    payoffSum += localPayoffSum;
}

double monteCarloOptionPricingParallel(double S0, double K, double r, double sigma, double T, int numSimulations, bool isCallOption, int numThreads) {
    std::vector<std::thread> threads;
    std::mutex mtx;
    double payoffSum = 0.0;
    int simulationsPerThread = numSimulations / numThreads;

    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(monteCarloWorker, S0, K, r, sigma, T,
                             simulationsPerThread, isCallOption, std::ref(payoffSum), std::ref(mtx));
    }

    for (auto& th : threads) {
        th.join();
    }

    double averagePayoff = payoffSum / static_cast<double>(simulationsPerThread * numThreads);
    return std::exp(-r * T) * averagePayoff;
}

int main() {
    double S0 = 100.0, K = 100.0, r = 0.05, sigma = 0.2, T = 1.0;
    int numSimulations = 1000000; 
    int numThreads = std::thread::hardware_concurrency(); // Use max hardware threads

    std::cout << "Threads used: " << numThreads << std::endl;

    double callPrice = monteCarloOptionPricingParallel(S0, K, r, sigma, T, numSimulations, true, numThreads);
    double putPrice = monteCarloOptionPricingParallel(S0, K, r, sigma, T, numSimulations, false, numThreads);

    std::cout << "European Call Option Price: " << callPrice << std::endl;
    std::cout << "European Put Option Price: " << putPrice << std::endl;

    return 0;
}
