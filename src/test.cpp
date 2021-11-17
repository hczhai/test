
#include "main.hpp"
#include "matrix_functions.hpp"
#include "iterative_matrix_functions.hpp"
#include "gtest/gtest.h"

// using namespace xtest;
using namespace block2;

class TestX : public ::testing::Test
{
protected:
    static const int n_tests_x = 1000000;
    static const int n_tests = 100;
    struct MatMul
    {
        ComplexMatrixRef a;
        MatMul(const ComplexMatrixRef &a) : a(a) {}
        void operator()(const ComplexMatrixRef &b, const ComplexMatrixRef &c)
        {
            ComplexMatrixFunctions::multiply(a, false, b, false, c, 1.0, 0.0);
        }
    };
    size_t isize = 1L << 24;
    size_t dsize = 1L << 28;
    void SetUp() override
    {
        Random::rand_seed(1234);
        frame_() = make_shared<DataFrame>(isize, dsize, "nodex");
    }
    void TearDown() override
    {
        frame_()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_()->used == 0);
        frame_() = nullptr;
    }
};

TEST_F(TestX, TestLSQR)
{
    for (int i = 0; i < n_tests * 100000; i++)
    {
        MKL_INT m = Random::rand_int(1, 7);
        MKL_INT n = 1;
        int nmult = 0, niter = 0;
        double eta = 0.05;
        MatrixRef ra(dalloc_()->allocate(m * m), m, m);
        MatrixRef rax(dalloc_()->allocate(m * m), m, m);
        MatrixRef rb(dalloc_()->allocate(n * m), m, n);
        MatrixRef rbg(dalloc_()->allocate(n * m), m, n);
        ComplexMatrixRef a(dalloc_()->complex_allocate(m * m), m, m);
        ComplexMatrixRef af(dalloc_()->complex_allocate(m * m), m, m);
        ComplexMatrixRef b(dalloc_()->complex_allocate(n * m), m, n);
        ComplexMatrixRef x(dalloc_()->complex_allocate(n * m), m, n);
        ComplexMatrixRef xg(dalloc_()->complex_allocate(n * m), m, n);
        Random::fill<double>(ra.data, ra.size());
        Random::fill<double>(rax.data, rax.size());
        Random::fill<double>(rb.data, rb.size());
        a.clear();
        b.clear();
        MatrixFunctions::multiply(rax, false, rax, true, ra, 1.0, 0.0);
        ComplexMatrixFunctions::fill_complex(a, ra, MatrixRef(nullptr, m, m));
        for (MKL_INT k = 0; k < n; k++)
            a(k, k) += complex<double>(0, eta);
        ComplexMatrixFunctions::fill_complex(b, rb, MatrixRef(nullptr, m, n));
        Random::fill<double>(rb.data, rb.size());
        ComplexMatrixFunctions::fill_complex(x, rb, MatrixRef(nullptr, m, n));
        Random::fill<double>(rb.data, rb.size());
        ComplexMatrixFunctions::fill_complex(x, MatrixRef(nullptr, m, n), rb);
        ComplexMatrixFunctions::copy(af, a);
        ComplexMatrixFunctions::conjugate(af);
        MatMul mop(a), rop(af);
        // hrl: Note: The input matrix can be highly illconditioned (cond~10^5)
        //      which causes problems for lsqr.
        //      It is important to have long maxiters and small atol.
        //      It may still fail in extreme situations,
        //      in particular when m ~ 300.
        complex<double> func = IterativeMatrixFunctions<complex<double>>::lsqr(
            mop, rop, ComplexDiagonalMatrix(nullptr, 0), x, b, nmult, niter,
            false, (shared_ptr<ParallelCommunicator<void>>)nullptr, 1E-8, 1E-7,
            0., 10000);
        ComplexMatrixFunctions::copy(xg, b);
        for (MKL_INT k = 0; k < m; k++)
            for (MKL_INT j = 0; j < m; j++)
                af(k, j) = a(j, k);
        ComplexMatrixFunctions::linear(af, xg.flip_dims());
        ComplexMatrixFunctions::extract_complex(xg, rbg,
                                                MatrixRef(nullptr, m, n));
        ComplexMatrixFunctions::extract_complex(x, rb,
                                                MatrixRef(nullptr, m, n));
        bool bba = MatrixFunctions::all_close(rbg, rb, 1E-3, 1E-3);
        ComplexMatrixFunctions::extract_complex(xg, MatrixRef(nullptr, m, n),
                                                rbg);
        ComplexMatrixFunctions::extract_complex(x, MatrixRef(nullptr, m, n),
                                                rb);
        bool bbb = MatrixFunctions::all_close(rbg, rb, 1E-3, 1E-3);
        if (!bba || !bbb)
        {
            cout << "not working : a = " << a << endl;
            cout << "not working : b = " << b << endl;
            cout << "not working : x-ref  = " << xg << endl;
            cout << "not working : x-lsqr = " << x << endl;
        }
        EXPECT_TRUE(bba && bbb);
        xg.deallocate();
        x.deallocate();
        b.deallocate();
        af.deallocate();
        a.deallocate();
        rbg.deallocate();
        rb.deallocate();
        rax.deallocate();
        ra.deallocate();
    }
}

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

// TEST_F(TestX, TestHarmonicDavidson)
// {
//     for (int i = 0; i < n_tests; i++)
//     {
//         MKL_INT n = Random::rand_int(3, 50);
//         MKL_INT k = min(n, (MKL_INT)Random::rand_int(1, 5));
//         if (k > n / 2)
//             k = n / 2;
//         int ndav = 0;
//         MatrixRef a(dalloc_()->allocate(n * n), n, n);
//         DiagonalMatrix aa(dalloc_()->allocate(n), n);
//         DiagonalMatrix ww(dalloc_()->allocate(n), n);
//         vector<MatrixRef> bs(k, MatrixRef(nullptr, n, 1));
//         Random::fill_rand_double(a.data, a.size());
//         for (MKL_INT ki = 0; ki < n; ki++)
//         {
//             for (MKL_INT kj = 0; kj < ki; kj++)
//                 a(kj, ki) = a(ki, kj);
//             aa(ki, ki) = a(ki, ki);
//         }
//         for (int i = 0; i < k; i++)
//         {
//             bs[i].allocate();
//             bs[i].clear();
//             bs[i].data[i] = 1;
//         }
//         MatMul mop(a);
//         double shift = 0.1;
//         DavidsonTypes davidson_type =

//             Random::rand_int(0, 2)
//                 ? DavidsonTypes::HarmonicLessThan | DavidsonTypes::NoPrecond
//                 : DavidsonTypes::HarmonicGreaterThan | DavidsonTypes::NoPrecond;
//         vector<double> vw = MatrixFunctions::harmonic_davidson(
//             mop, aa, bs, shift, davidson_type, ndav, false,
//             (shared_ptr<ParallelCommunicator<void>>)nullptr, 1E-8, n * k * 100,
//             -1, 2, 30);
//         ASSERT_EQ((int)vw.size(), k);
//         DiagonalMatrix w(&vw[0], k);
//         MatrixFunctions::eigs(a, ww);
//         vector<int> eigval_idxs(ww.size());
//         for (int i = 0; i < (int)ww.size(); i++)
//             eigval_idxs[i] = i;
//         if (davidson_type & DavidsonTypes::CloseTo)
//             sort(eigval_idxs.begin(), eigval_idxs.end(),
//                  [&ww, shift](int i, int j)
//                  {
//                      return abs(ww.data[i] - shift) < abs(ww.data[j] - shift);
//                  });
//         else if (davidson_type & DavidsonTypes::LessThan)
//             sort(eigval_idxs.begin(), eigval_idxs.end(),
//                  [&ww, shift](int i, int j)
//                  {
//                      if ((shift >= ww.data[i]) != (shift >= ww.data[j]))
//                          return shift >= ww.data[i];
//                      else if (shift >= ww.data[i])
//                          return shift - ww.data[i] < shift - ww.data[j];
//                      else
//                          return ww.data[i] - shift > ww.data[j] - shift;
//                  });
//         else if (davidson_type & DavidsonTypes::GreaterThan)
//             sort(eigval_idxs.begin(), eigval_idxs.end(),
//                  [&ww, shift](int i, int j)
//                  {
//                      if ((shift > ww.data[i]) != (shift > ww.data[j]))
//                          return shift > ww.data[j];
//                      else if (shift > ww.data[i])
//                          return shift - ww.data[i] > shift - ww.data[j];
//                      else
//                          return ww.data[i] - shift < ww.data[j] - shift;
//                  });
//         // last root may be inaccurate (rare)
//         for (int i = 0; i < k - 1; i++)
//             ASSERT_LT(abs(ww.data[eigval_idxs[i]] - vw[i]), 1E-6);
//         for (int i = 0; i < k - 1; i++)
//             ASSERT_TRUE(
//                 MatrixFunctions::all_close(
//                     bs[i], MatrixRef(a.data + a.n * eigval_idxs[i], a.n, 1),
//                     1E-3, 1E-3) ||
//                 MatrixFunctions::all_close(
//                     bs[i], MatrixRef(a.data + a.n * eigval_idxs[i], a.n, 1),
//                     1E-3, 1E-3, -1.0));
//         for (int i = k - 1; i >= 0; i--)
//             bs[i].deallocate();
//         ww.deallocate();
//         aa.deallocate();
//         a.deallocate();
//     }
// }
