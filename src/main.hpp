
#include <bits/stdc++.h>

using namespace std;

namespace xtest
{

    // Random number generator
    struct Random
    {
        static mt19937 &rng()
        {
            static mt19937 _rng;
            return _rng;
        }
        static void rand_seed(unsigned i = 0)
        {
            rng() = mt19937(i ? i
                              : (unsigned)chrono::steady_clock::now()
                                    .time_since_epoch()
                                    .count());
        }
        // return a integer in [a, b)
        static int rand_int(int a, int b)
        {
            assert(b > a);
            return uniform_int_distribution<int>(a, b - 1)(rng());
        }
        // return a double in [a, b)
        static double rand_double(double a = 0, double b = 1)
        {
            assert(b > a);
            return uniform_real_distribution<double>(a, b)(rng());
        }
        static void fill_rand_float(float *data, size_t n, float a = 0,
                                    float b = 1)
        {
            uniform_real_distribution<float> distr(a, b);
            for (size_t i = 0; i < n; i++)
                data[i] = distr(rng());
        }
        static void fill_rand_double(double *data, size_t n, double a = 0,
                                     double b = 1)
        {
            uniform_real_distribution<double> distr(a, b);
            for (size_t i = 0; i < n; i++)
                data[i] = distr(rng());
        }
    };

    struct XTest
    {
        int a, b;
        XTest(int a, int b) : a(a), b(b) {}
        XTest operator+(XTest other) const
        {
            return XTest(a + other.a, b + other.b);
        }
    };

}