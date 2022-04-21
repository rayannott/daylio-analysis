#include <iostream>
#include <stdio.h>
#include <stdexcept>
#include <cmath>
#include <fstream>
#include <string>
using namespace std;
#define pi 3.1415926535

long double alpha(const   double& x) {
    return sqrt(x / (1 + x));
}


// функция считает саха-классик // тут всё в эВ
long double Saha(long double P, long double T, long double g_i, long double I) {
    long double _h = 0.659 * 1e-15;
    long double m_e = 0.510 * 1e6;
    long double g_A = 1;
    long double K_B = 8.617 * 1e-5;
    return exp(-I / (K_B * T)) * (2 * g_i / g_A) * pow((m_e / (2 * pi * _h * _h)), (3 / 2)) * pow((K_B * T), (5 / 2)) / P;
}

// функция записывает результат в файлик
void Saha_txt(const string fileName, long double g_i, long double I) {
    double p = 1;
    ofstream out;
    out.open(fileName);
    long double P = p * 1e5 * 6.25 * 1e18;
    if (out.is_open()) {
        for (int T = 1000; T <= 20000; T += 100) {

            long double S_K = Saha(P, T, g_i, I);
            out << alpha(S_K) << "\n";
        }
    }
}

int main() {
    // это сгс, для красоты надо переделать в эВ
    long double m_e = 9.1 * 1e-28;
    long double K_Bsgs = 1.38067 * 1e-16;
    long double sigma_e_ar = 1e-18;
    long double sigma_e_k = 1e-15;
    long double e = 4.8 * 1e-10;

    double p = 1;

    ofstream out;
    out.open("sigma(T).txt");
    long double P = p * 1e5 * 6.25 * 1e18;

    if (out.is_open()) {
        for (int T = 1000; T <= 10000; T += 100) {
            long double S;

            long double g_i = 1;
            long double I = 4.34;
            S = Saha(P, T, g_i, I);


            long double v_t = sqrt(3 * T * K_Bsgs / m_e);
            long double a = sigma_e_ar / sigma_e_k;
            long double sigma = e * e * sqrt(S * K_Bsgs * T / (p * 1e6)) / (m_e * v_t * sigma_e_k * 2 * sqrt(a));
            //if (T == 2300)
                //cout << "Conductivity at T=2300 sigma = " << sigma / (9 * 1e9) << "\n" << "Optimal K/Ar ratio x = " << a;

            out << sigma / (9 * 1e9) << "\n";

        }
    }

    // saha calium
    long double I_K = 4.34;
    Saha_txt("alpha_k.txt", 1, I_K);

    // saha argon
    long double I_Ar = 15.76;
    Saha_txt("alpha_Ar.txt", 6, I_Ar);

    out.open("sigma(x).txt");
    P = 1 * 1e5 * 6.25 * 1e18;
    
        for (long double x = 0.0001; x <= 0.01; x += 0.0001) {
            long double S;

            long double g_i = 1;
            long double I = 4.34;
            S = Saha(P, 2300, g_i, I);

            long double v_t = sqrt(3 * 2300 * K_Bsgs / m_e);
            long double a = sigma_e_ar / sigma_e_k;
            long double sigma = (sqrt(x) / (x + a)) * e * e * sqrt(S * K_Bsgs * 2300 / (p * 1e6)) / (m_e * v_t * sigma_e_k);

            out << sigma / (9 * 1e9) << "\n";
        }
    
    return 0;
}

