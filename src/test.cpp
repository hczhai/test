
#include "main.hpp"
#include "matrix_functions.hpp"
#include "gtest/gtest.h"

// using namespace xtest;
using namespace block2;

class TestX : public ::testing::Test
{
protected:
    static const int n_tests_x = 1000000;
    static const int n_tests = 100;
    struct MatMul {
        MatrixRef a;
        MatMul(const MatrixRef &a) : a(a) {}
        void operator()(const MatrixRef &b, const MatrixRef &c) {
            MatrixFunctions::multiply(a, false, b, false, c, 1.0, 0.0);
        }
    };
    size_t isize = 1L << 24;
    size_t dsize = 1L << 28;
    void SetUp() override {
        Random::rand_seed(0);
        frame_() = make_shared<DataFrame>(isize, dsize, "nodex");
    }
    void TearDown() override {
        frame_()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_()->used == 0);
        frame_() = nullptr;
    }
};

// TEST_F(TestX, TestXX)
// {
//     for (int i = 0; i < n_tests_x; i++)
//     {
//         XTest xa(Random::rand_int(-99999, 99999), Random::rand_int(-99999, 99999));
//         XTest xb(Random::rand_int(-99999, 99999), Random::rand_int(-99999, 99999));
//         XTest xc = xa + xb;
//         EXPECT_EQ(xa.a + xb.a, xc.a);
//         EXPECT_EQ(xa.b + xb.b, xc.b);
//     }
// }



TEST_F(TestX, TestHarmonicDavidson) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT n = Random::rand_int(3, 50);
        MKL_INT k = min(n, (MKL_INT)Random::rand_int(1, 5));
        if (k > n / 2)
            k = n / 2;
        int ndav = 0;
        MatrixRef a(dalloc_()->allocate(n * n), n, n);
        DiagonalMatrix aa(dalloc_()->allocate(n), n);
        DiagonalMatrix ww(dalloc_()->allocate(n), n);
        vector<MatrixRef> bs(k, MatrixRef(nullptr, n, 1));
        Random::fill_rand_double(a.data, a.size());
        for (MKL_INT ki = 0; ki < n; ki++) {
            for (MKL_INT kj = 0; kj < ki; kj++)
                a(kj, ki) = a(ki, kj);
            aa(ki, ki) = a(ki, ki);
        }
        for (int i = 0; i < k; i++) {
            bs[i].allocate();
            bs[i].clear();
            bs[i].data[i] = 1;
        }
        MatMul mop(a);
        double shift = 0.1;
        DavidsonTypes davidson_type =
            
            Random::rand_int(0, 2)
                ? DavidsonTypes::HarmonicLessThan | DavidsonTypes::NoPrecond
                : DavidsonTypes::HarmonicGreaterThan | DavidsonTypes::NoPrecond;
        vector<double> vw = MatrixFunctions::harmonic_davidson(
            mop, aa, bs, shift, davidson_type, ndav, false,
            (shared_ptr<ParallelCommunicator<void>>)nullptr, 1E-8, n * k * 100,
            -1, 2, 30);
        ASSERT_EQ((int)vw.size(), k);
        DiagonalMatrix w(&vw[0], k);
        MatrixFunctions::eigs(a, ww);
        vector<int> eigval_idxs(ww.size());
        for (int i = 0; i < (int)ww.size(); i++)
            eigval_idxs[i] = i;
        if (davidson_type & DavidsonTypes::CloseTo)
            sort(eigval_idxs.begin(), eigval_idxs.end(),
                 [&ww, shift](int i, int j) {
                     return abs(ww.data[i] - shift) < abs(ww.data[j] - shift);
                 });
        else if (davidson_type & DavidsonTypes::LessThan)
            sort(eigval_idxs.begin(), eigval_idxs.end(),
                 [&ww, shift](int i, int j) {
                     if ((shift >= ww.data[i]) != (shift >= ww.data[j]))
                         return shift >= ww.data[i];
                     else if (shift >= ww.data[i])
                         return shift - ww.data[i] < shift - ww.data[j];
                     else
                         return ww.data[i] - shift >= ww.data[j] - shift;
                 });
        else if (davidson_type & DavidsonTypes::GreaterThan)
            sort(eigval_idxs.begin(), eigval_idxs.end(),
                 [&ww, shift](int i, int j) {
                     if ((shift > ww.data[i]) != (shift > ww.data[j]))
                         return shift > ww.data[j];
                     else if (shift > ww.data[i])
                         return shift - ww.data[i] > shift - ww.data[j];
                     else
                         return ww.data[i] - shift <= ww.data[j] - shift;
                 });
        // last root may be inaccurate (rare)
        for (int i = 0; i < k - 1; i++)
            ASSERT_LT(abs(ww.data[eigval_idxs[i]] - vw[i]), 1E-6);
        for (int i = 0; i < k - 1; i++)
            ASSERT_TRUE(
                MatrixFunctions::all_close(
                    bs[i], MatrixRef(a.data + a.n * eigval_idxs[i], a.n, 1),
                    1E-3, 1E-3) ||
                MatrixFunctions::all_close(
                    bs[i], MatrixRef(a.data + a.n * eigval_idxs[i], a.n, 1),
                    1E-3, 1E-3, -1.0));
        for (int i = k - 1; i >= 0; i--)
            bs[i].deallocate();
        ww.deallocate();
        aa.deallocate();
        a.deallocate();
    }
}