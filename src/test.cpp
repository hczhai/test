
#include "main.hpp"
#include "gtest/gtest.h"

using namespace xtest;

class TestX : public ::testing::Test
{
protected:
    static const int n_tests = 1000000;
    void SetUp() override { Random::rand_seed(0); }
    void TearDown() override {}
};

TEST_F(TestX, TestXX)
{
    for (int i = 0; i < n_tests; i++)
    {
        XTest xa(Random::rand_int(-99999, 99999), Random::rand_int(-99999, 99999));
        XTest xb(Random::rand_int(-99999, 99999), Random::rand_int(-99999, 99999));
        XTest xc = xa + xb;
        EXPECT_EQ(xa.a + xb.a, xc.a);
        EXPECT_EQ(xa.b + xb.b, xc.b);
    }
}